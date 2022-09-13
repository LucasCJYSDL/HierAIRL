import torch
import torch.nn.functional as F
from .option_policy import OptionPolicy, Policy
from .option_discriminator import OptionDiscriminator, Discriminator
from utils.config import Config
from utils.model_util import clip_grad_norm_

class AIRL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(AIRL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.device = torch.device(config.device)
        self.mini_bs = config.mini_batch_size
        lr = config.optimizer_lr_discriminator

        self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = Policy(config, dim_s=dim_s, dim_a=dim_a)
        self.criterion = torch.nn.BCELoss()

        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=3.e-5) # only update the disc and fix the policy

        self.to(self.device)

    def airl_reward(self, s, a):
        log_sa = self.policy.log_prob_action(s, a) # (N, 1)
        log_sa = log_sa.detach().clone()
        f = self.discriminator.get_unnormed_d(s, a) # (N, 1)
        exp_f = torch.exp(f)
        # d = (exp_f / (exp_f + 1.0)).detach().clone()
        d = (exp_f / (exp_f + torch.exp(log_sa))).detach().clone() # (N, 1)
        # reward = torch.log(d + 1e-6) - torch.log((1-d) + 1e-6)
        # reward = torch.log(1 - d + 1e-6) - torch.log(d + 1e-6)
        # reward = - torch.log(d + 1e-6)
        # reward = F.softplus(f, beta=-1)
        reward = d
        # print("here: ", reward)

        return reward

    def step(self, sample_sar, demo_sar, n_step=10):
        sp = torch.cat([s for s, a, r in sample_sar], dim=0)
        se = torch.cat([s for s, a, r in demo_sar], dim=0)
        ap = torch.cat([a for s, a, r in sample_sar], dim=0)
        ae = torch.cat([a for s, a, r in demo_sar], dim=0)
        # huge difference compared with gail
        tp = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the generated state-action pairs
        te = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device) # label for the expert state-action pairs

        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                sp_b, ap_b, tp_b = sp[ind_p], ap[ind_p], tp[:ind_p.size(0)]
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                se_b, ae_b, te_b = se[ind_e], ae[ind_e], te[:ind_p.size(0)]

                for _ in range(3):
                    # for the generated data
                    f_b = self.discriminator.get_unnormed_d(sp_b, ap_b)
                    log_sa_b = self.policy.log_prob_action(sp_b, ap_b)
                    log_sa_b = log_sa_b.detach().clone()
                    exp_f_b = torch.exp(f_b)

                    d_b = exp_f_b / (exp_f_b + torch.exp(log_sa_b)) #  a prob between 0. to 1.
                    d_b = torch.clamp(d_b, min=1e-3, max=1 - 1e-3)
                    loss_b = self.criterion(d_b, tp_b)
                    # for the expert data
                    f_e = self.discriminator.get_unnormed_d(se_b, ae_b)
                    log_sa_e = self.policy.log_prob_action(se_b, ae_b)
                    log_sa_e = log_sa_e.detach().clone()
                    exp_f_e = torch.exp(f_e)
                    d_e = exp_f_e / (exp_f_e + torch.exp(log_sa_e))
                    d_e = torch.clamp(d_e, min=1e-3, max=1 - 1e-3)
                    loss_e = self.criterion(d_e, te_b)
                    loss = loss_b + loss_e
                    loss += self.discriminator.gradient_penalty(sp_b, ap_b, lam=10.)
                    loss += self.discriminator.gradient_penalty(se_b, ae_b, lam=10.)

                    self.optim.zero_grad()
                    loss.backward()
                    # for p in self.discriminator.parameters():
                    #     print("before: ", p.data.norm(2))
                    # clip_grad_norm_(self.discriminator.parameters(), max_norm=20, norm_type=2)
                    # for p in self.discriminator.parameters():
                    #     print("after: ", p.data.norm(2))
                    self.optim.step()



    def convert_demo(self, demo_sa):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                r_fake_array = self.airl_reward(s_array, a_array)
                out_sample.append((s_array, a_array, r_fake_array))
                r_sum_avg += r_fake_array.sum().item()
            r_sum_avg /= len(demo_sa)
        return out_sample, r_sum_avg

    def convert_sample(self, sample_sar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array, r_real_array in sample_sar:
                r_fake_array = self.airl_reward(s_array, a_array)
                out_sample.append((s_array, a_array, r_fake_array))
                r_sum_avg += r_real_array.sum().item()
            r_sum_avg /= len(sample_sar)
        return out_sample, r_sum_avg


class OptionAIRL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(OptionAIRL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_c = config.dim_c
        self.with_c = config.use_c_in_discriminator

        self.mini_bs = config.mini_batch_size
        self.use_d_info_gail = config.use_d_info_gail
        self.device = torch.device(config.device)

        assert self.with_c and not self.use_d_info_gail

        self.discriminator = OptionDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = OptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)
        self.criterion = torch.nn.BCELoss()

        self.optim = torch.optim.Adam(self.discriminator.parameters(), weight_decay=3.e-5)
        self.to(self.device)


    def airl_reward(self, s, c_1, a, c):
        f = self.discriminator.get_unnormed_d(s, c_1, a, c) # (N, 1)
        log_sc = self.policy.log_prob_option(s, c_1, c).detach().clone() # (N, 1)
        log_sa = self.policy.log_prob_action(s, c, a).detach().clone() # (N, 1)
        sca = torch.exp(log_sc) * torch.exp(log_sa)
        exp_f = torch.exp(f)
        d = (exp_f / (exp_f + sca)).detach().clone()
        reward = torch.log(d + 1e-6) - torch.log((1-d) + 1e-6)
        return reward


    def step(self, sample_scar, demo_scar, n_step=10):
        sp = torch.cat([s for s, c, a, r in sample_scar], dim=0)
        se = torch.cat([s for s, c, a, r in demo_scar], dim=0)
        c_1p = torch.cat([c[:-1] for s, c, a, r in sample_scar], dim=0)
        c_1e = torch.cat([c[:-1] for s, c, a, r in demo_scar], dim=0)
        cp = torch.cat([c[1:] for s, c, a, r in sample_scar], dim=0)
        ce = torch.cat([c[1:] for s, c, a, r in demo_scar], dim=0)
        ap = torch.cat([a for s, c, a, r in sample_scar], dim=0)
        ae = torch.cat([a for s, c, a, r in demo_scar], dim=0)
        # huge difference compared with gail
        tp = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device)  # label for the generated state-action pairs
        te = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device)  # label for the expert state-action pairs

        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                sp_b, cp_1b, ap_b, cp_b, tp_b = sp[ind_p], c_1p[ind_p], ap[ind_p], cp[ind_p], tp[:ind_p.size(0)]
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                se_b, ce_1b, ae_b, ce_b, te_b = se[ind_e], c_1e[ind_e], ae[ind_e], ce[ind_e], te[:ind_p.size(0)]

                s_array = torch.cat((sp_b, se_b), dim=0)
                a_array = torch.cat((ap_b, ae_b), dim=0)
                c_1array = torch.cat((cp_1b, ce_1b), dim=0)
                c_array = torch.cat((cp_b, ce_b), dim=0)
                t_array = torch.cat((tp_b, te_b), dim=0)

                for _ in range(1):

                    f = self.discriminator.get_unnormed_d(s_array, c_1array, a_array, c_array)
                    exp_f = torch.exp(f)
                    log_sc = self.policy.log_prob_option(s_array, c_1array, c_array).detach().clone()
                    log_sa = self.policy.log_prob_action(s_array, c_array, a_array).detach().clone()
                    sca = torch.exp(log_sc) * torch.exp(log_sa)
                    d = exp_f / (exp_f + sca)
                    loss = self.criterion(d, t_array)
                    loss += self.discriminator.gradient_penalty(s_array, a_array, c_1array, c_array, lam=10.)
                    # print("2: ", loss)

                    self.optim.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.discriminator.parameters(), max_norm=5, norm_type=2)
                    self.optim.step()


    def convert_demo(self, demo_sa):
        with torch.no_grad(): # important
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                assert self.with_c
                c_array, _ = self.policy.viterbi_path(s_array, a_array)
                # for option_airl, it does not have a posterior,
                # so it has to estimate the c_array with its hier policy like option_gail does
                r_array = self.airl_reward(s_array, c_array[:-1], a_array, c_array[1:])
                out_sample.append((s_array, c_array, a_array, r_array))
                r_sum_avg += r_array.sum().item()
            r_sum_avg /= len(demo_sa)
        return out_sample, r_sum_avg


    def convert_sample(self, sample_scar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, c_array, a_array, r_real_array in sample_scar:
                r_fake_array = self.airl_reward(s_array, c_array[:-1], a_array, c_array[1:])
                out_sample.append((s_array, c_array, a_array, r_fake_array))
                r_sum_avg += r_real_array.sum().item()
            r_sum_avg /= len(sample_scar)
        return out_sample, r_sum_avg



