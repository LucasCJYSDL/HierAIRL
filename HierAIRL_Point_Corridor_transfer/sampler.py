import torch
import time
from copy import deepcopy
from torch.multiprocessing import Process, Pipe, Lock, Value
from model.option_policy import OptionPolicy
from model.MHA_option_policy_critic import MHAOptionPolicy
from utils.common_utils import set_seed
from envir import mujoco_maze


__all__ = ["Sampler"]

# rlbench: 4096, 1t, 135s; 2t, 79s; 4t, 51s; 6t, 51s
# mujoco: 5000, 1t, 7.2s; 2t, 5.6s; 4t, 4.2s; 6t, 4.2s


class _sQueue(object):
    def __init__(self, pipe_rw, r_lock, w_lock):
        self.rlock = r_lock
        self.wlock = w_lock
        self.pipe_rw = pipe_rw

    def __del__(self):
        self.pipe_rw.close()

    def get(self, time_out=0.):
        d = None
        if self.pipe_rw.poll(time_out):
            with self.rlock:
                d = self.pipe_rw.recv()
        return d

    def send(self, d):
        with self.wlock:
            self.pipe_rw.send(d)


def pipe_pair():
    p_lock = Lock()
    c_lock = Lock()
    pipe_c, pipe_p = Pipe(duplex=True)
    child_q = _sQueue(pipe_c, p_lock, c_lock)
    parent_q = _sQueue(pipe_p, c_lock, p_lock)
    return child_q, parent_q


def option_loop(env, policy, fixed):
    with torch.no_grad():
        a_array = []
        c_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        ct = torch.empty(1, 1, dtype=torch.long, device=policy.device).fill_(policy.dim_c)
        c_array.append(ct)
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            ct = policy.sample_option(st, ct, fixed=fixed).detach()
            at = policy.sample_action(st, ct, fixed=fixed).detach()
            s_array.append(st)
            c_array.append(ct)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            # env.render()
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        c_array = torch.cat(c_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
    return s_array, c_array, a_array, r_array


def loop(env, policy, fixed):
    with torch.no_grad():
        a_array = []
        s_array = []
        r_array = []
        s, done = env.reset(random=not fixed), False
        while not done:
            st = torch.as_tensor(s, dtype=torch.float32, device=policy.device).unsqueeze(0)
            at = policy.sample_action(st, fixed=fixed).detach()
            s_array.append(st)
            a_array.append(at)
            s, r, done = env.step(at.cpu().squeeze(dim=0).numpy())
            r_array.append(r)
        a_array = torch.cat(a_array, dim=0)
        s_array = torch.cat(s_array, dim=0)
        r_array = torch.as_tensor(r_array, dtype=torch.float32, device=policy.device).unsqueeze(dim=-1)
        # print(r_array.shape)
    return s_array, a_array, r_array


class _SamplerCommon(object):
    def __init__(self, seed, policy):
        self.device = policy.device

    def collect(self, policy_param, n_sample, fixed=False):
        raise NotImplementedError()

    def filter_demo(self, sa_array):
        print("No filters are adopted.")
        return sa_array



class _Sampler(_SamplerCommon):
    def __init__(self, seed, env, policy, n_thread=4, loop_func=None):
        super(_Sampler, self).__init__(seed, policy)
        self.counter = Value('i', 0)
        self.state = Value('i', n_thread)
        child_q, self.queue = pipe_pair()
        self.procs = [Process(target=self.worker, name=f"subproc_{seed}",
                              args=(seed, env, policy, loop_func, self.state, self.counter, child_q))
                      for _ in range(n_thread)]
        self.pids = []
        for p in self.procs:
            p.daemon = True
            p.start()
            self.pids.append(p.pid)

        while self.state.value > 0: # wait for all the workers to prepare well
            time.sleep(0.1)

    def collect(self, policy_param, n_sample, fixed=False):
        # n_sample <0 for number of trajectories, >0 for number of sa pairs
        for _ in self.procs: # send the msg from the parent node to the child node
            self.queue.send((policy_param, fixed))

        with self.state.get_lock(): # updating the value of state or counter needs to be done with in get_lock(); tell the workers to receive the pi parameters
            self.state.value = -len(self.procs)

        while self.state.value < 0: # wait for all the workers to receive and load the parameters
            time.sleep(0.1)

        with self.counter.get_lock():
            self.counter.value = n_sample

        with self.state.get_lock(): # tell the workers to start to collect counter.value datas
            self.state.value = len(self.procs)

        ret = []
        while self.state.value > 0: # wait until the workers to collect enough data
            d = self.queue.get(0.0001)
            while d is not None:
                traj = d
                ret.append(tuple(x.to(self.device) for x in traj))
                d = self.queue.get(0.0001)

        return ret

    def __del__(self):
        print(f"agent process is terminated, check if any subproc left: {self.pids}")
        for p in self.procs:
            p.terminate()

    @staticmethod
    def worker(seed: int, env, policy, loop_func, state: Value, counter: Value, queue: _sQueue):
        # state 0: idle, -n: init param, n: sampling
        set_seed(seed)

        env.init(display=False)
        with state.get_lock():
            state.value -= 1

        while True:
            while state.value >= 0: # wait for the policy parameters from collect(); starts from 0 everytime.
                time.sleep(0.1)

            d = None
            while d is None:
                d = queue.get(5)

            net_param, fixed = d
            policy.load_state_dict(net_param)

            with state.get_lock(): # all the works do this step, so the state-value will become 0
                state.value += 1

            while state.value <= 0: # wait for the amount of data to sample from collect()
                time.sleep(0.1)

            while state.value > 0: # wait for collecting enough data
                traj = loop_func(env, policy, fixed=fixed)
                with counter.get_lock():
                    if counter.value > 0:
                        queue.send(tuple(x.cpu() for x in traj))
                        counter.value -= traj[0].size(0)
                        if counter.value <= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0
                    elif counter.value < 0:
                        queue.send(tuple(x.cpu() for x in traj))
                        counter.value += 1
                        if counter.value >= 0:
                            counter.value = 0
                            with state.get_lock():
                                state.value = 0


class _SamplerSS(_SamplerCommon):
    def __init__(self, seed, env, policy, n_thread=1, loop_func=None):
        super(_SamplerSS, self).__init__(seed, policy)
        if n_thread > 1:
            print(f"Warning: you are using single thread sampler, despite n_thread={n_thread}")
        self.env = deepcopy(env)
        self.env.init(display=False)
        self.policy = deepcopy(policy)
        self.loop_func = loop_func

    def collect(self, policy_param, n_sample, fixed=False):
        self.policy.load_state_dict(policy_param)
        counter = n_sample
        rets = []
        if counter > 0:
            while counter > 0:
                traj = self.loop_func(self.env, self.policy, fixed=fixed)
                rets.append(traj)
                counter -= traj[0].size(0)
        else:
            while counter < 0:
                traj = self.loop_func(self.env, self.policy, fixed=fixed)
                rets.append(traj)
                counter += 1
        return rets


def Sampler(seed, env, policy, n_thread=4) -> _SamplerCommon:
    if isinstance(policy, OptionPolicy) or isinstance(policy, MHAOptionPolicy):
        loop_func = option_loop
    else:
        loop_func = loop
    class_m = _Sampler if n_thread > 1 else _SamplerSS
    return class_m(seed, env, policy, n_thread, loop_func)


if __name__ == "__main__":
    from torch.multiprocessing import set_start_method
    set_start_method("spawn")
