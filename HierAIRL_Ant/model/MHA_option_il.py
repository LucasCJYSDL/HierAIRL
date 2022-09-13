import torch
import torch.nn.functional as F
from .MHA_option_policy_critic import MHAOptionPolicy
from .option_policy import OptionPolicy
from .option_discriminator import OptionDiscriminator
from utils.config import Config
from utils.model_util import GRUPosterior
from utils.model_util import clip_grad_norm_


class MHAOptionAIRL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(MHAOptionAIRL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_c = config.dim_c

        self.mini_bs = config.mini_batch_size
        self.device = torch.device(config.device)
        self.use_posterior = config.use_posterior
        self.gru_training_iters = config.gru_training_iterations
        self.gru_include_action = config.gru_include_action
        self.alpha_2 = config.lambda_entropy_option
        self.use_posterior_sampling = config.use_posterior_sampling

        self.discriminator = OptionDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        if config.use_MHA_policy:
            print("Using the policy network with MHA......")
            self.policy = MHAOptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)
        else:
            print("Using the MLP policy network......")
            self.policy = OptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)

        if self.use_posterior:
            gru_input_dim = self.dim_s + self.dim_c + 1
            if self.gru_include_action:
                gru_input_dim += self.dim_a

            self.posterior = GRUPosterior(gru_input_dim, config.gru_hid_dim, self.dim_c, config.n_gru_layers, config.gru_dropout)
            self.gru_optim = torch.optim.Adam(self.posterior.parameters(), weight_decay=1.e-3)

        self.criterion = torch.nn.BCELoss()
        self.optim = torch.optim.Adam(self.discriminator.parameters(), weight_decay=1.e-3) # 1e-3
        self.to(self.device)


    def airl_reward(self, s, c_1, a, c):
        f = self.discriminator.get_unnormed_d(s, c_1, a, c) # (N, 1)
        log_sc = self.policy.log_prob_option(s, c_1, c).detach().clone() # (N, 1)
        log_sa = self.policy.log_prob_action(s, c, a).detach().clone() # (N, 1)
        sca = torch.exp(log_sc) * torch.exp(log_sa)
        exp_f = torch.exp(f)
        d = (exp_f / (exp_f + 1.0)).detach().clone()
        reward = - torch.log((1-d) + 1e-8)

        if self.use_posterior: # TODO: add model.eval()
            next_s = s[1:]
            cur_a = a[:-1]
            pre_opt = c_1[:-1]
            target_opt = c[:-1]

            onehot_opt = F.one_hot(pre_opt.squeeze(-1), num_classes=self.dim_c + 1)
            if self.gru_include_action:
                gru_input = torch.cat([next_s, cur_a, onehot_opt], dim=-1)
            else:
                gru_input = torch.cat([next_s, onehot_opt], dim=-1)
            gru_input = gru_input.unsqueeze(1)  # batch_size is 1; no gradient info
            gru_output = self.posterior(gru_input)
            gru_logp_array = F.log_softmax(gru_output, dim=-1)
            gru_logp = gru_logp_array.gather(dim=-1, index=target_opt)

            gru_logp = torch.cat([gru_logp, torch.zeros((1, 1), dtype=torch.float32).to(gru_logp.device)], dim=0).detach().clone()
            # print("reward: ", reward.mean(), 'gru_logp: ', gru_logp.mean())
            reward = reward + self.alpha_2 * gru_logp # note that the entropy term will be included in the PPO part later

        return reward


    def step(self, sample_scar, demo_scar, n_step=10):
        # Posterior training
        if self.use_posterior:
            print("Training the posterior......")
            for _ in range(self.gru_training_iters):
                for s, c, a, r in sample_scar:
                    next_s = s[1:]
                    cur_a = a[:-1]
                    pre_opt = c[:-2]
                    target_opt = c[1:-1]
                    onehot_opt = F.one_hot(pre_opt.squeeze(-1), num_classes=self.dim_c+1)
                    if self.gru_include_action:
                        gru_input = torch.cat([next_s, cur_a, onehot_opt], dim=-1)
                    else:
                        gru_input = torch.cat([next_s, onehot_opt], dim=-1)
                    gru_input = gru_input.unsqueeze(1) # batch_size is 1; no gradient info
                    gru_output = self.posterior(gru_input) # (seq_len, dim_c)
                    # start training
                    gru_logp_array = F.log_softmax(gru_output, dim=-1) # (seq_len, dim_c)
                    gru_logp = gru_logp_array.gather(dim=-1, index=target_opt) # (seq_len, 1)
                    gru_loss = -torch.mean(gru_logp)

                    self.gru_optim.zero_grad()
                    gru_loss.backward()
                    self.gru_optim.step()

                print("GRU Loss: ", gru_loss.detach().clone().item())

        # Discriminator training
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

                for _ in range(3):

                    f = self.discriminator.get_unnormed_d(s_array, c_1array, a_array, c_array)
                    exp_f = torch.exp(f)
                    log_sc = self.policy.log_prob_option(s_array, c_1array, c_array).detach().clone()
                    log_sa = self.policy.log_prob_action(s_array, c_array, a_array).detach().clone()
                    sca = torch.exp(log_sc) * torch.exp(log_sa)
                    d = exp_f / (exp_f + 1.0)
                    loss = self.criterion(d, t_array)

                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()


    def convert_demo(self, demo_sa):
        with torch.no_grad(): # important
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                if not self.use_posterior:
                    c_array, _ = self.policy.viterbi_path(s_array, a_array)
                    # for option_airl, it does not have a posterior,
                    # so it has to estimate the c_array with its hier policy like option_gail does
                else:
                    if not self.use_posterior_sampling:
                        c_array, _ = self.policy.viterbi_path(s_array, a_array)
                    else: # of high variance, but very quick
                        print("Generating the option code sequence with the posterior......")
                        seq_len = int(s_array.size(0))
                        c_array = torch.zeros(s_array.size(0) + 1, 1, dtype=torch.long, device=self.device)
                        c_array[0] = self.dim_c
                        hidden = self.posterior.init_hidden()
                        for i in range(1, seq_len):
                            pre_opt = F.one_hot(c_array[i-1], num_classes=self.dim_c + 1)
                            next_s = s_array[i].unsqueeze(0)
                            cur_a = a_array[i-1].unsqueeze(0)
                            if self.gru_include_action:
                                gru_input = torch.cat([next_s, cur_a, pre_opt], dim=-1)
                            else:
                                gru_input = torch.cat([next_s, pre_opt], dim=-1)

                            gru_input = gru_input.unsqueeze(1)  # batch_size is 1; no gradient info
                            gru_output, hidden = self.posterior.forward_step(gru_input, hidden) # no grad info
                            gru_logp_array = F.log_softmax(gru_output, dim=-1)
                            # it's not appropriate to use argmax, since it will be greedy and can't guarantee the optimism
                            # use gumbel softmax to be in line with the high-level policy
                            opt = F.gumbel_softmax(gru_logp_array, hard=False).multinomial(1).long()  # (1, 1)
                            c_array[i] = opt
                        # given that we don't have S_T, we have to use the option_policy to sample the C_T
                        c_array[-1] = self.policy.sample_option(s_array[-1].unsqueeze(0), c_array[-2].unsqueeze(0))


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


class MHAOptionGAIL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(MHAOptionGAIL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_c = config.dim_c
        self.mini_bs = config.mini_batch_size
        self.device = torch.device(config.device)
        self.use_posterior = config.use_posterior
        self.gru_training_iters = config.gru_training_iterations
        self.gru_include_action = config.gru_include_action
        self.alpha_2 = config.lambda_entropy_option
        self.use_posterior_sampling = config.use_posterior_sampling

        self.discriminator = OptionDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        if config.use_MHA_policy:
            print("Using the policy network with MHA......")
            self.policy = MHAOptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)
        else:
            print("Using the MLP policy network......")
            self.policy = OptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)

        if self.use_posterior:
            gru_input_dim = self.dim_s + self.dim_c + 1
            if self.gru_include_action:
                gru_input_dim += self.dim_a

            self.posterior = GRUPosterior(gru_input_dim, config.gru_hid_dim, self.dim_c, config.n_gru_layers,
                                          config.gru_dropout)
            self.gru_optim = torch.optim.Adam(self.posterior.parameters(), weight_decay=1.e-3)

        self.optim = torch.optim.Adam(self.discriminator.parameters(), weight_decay=1.e-3)
        self.to(self.device)

    def gail_reward(self, s, c_1, a, c):
        d = self.discriminator.get_unnormed_d(s, c_1, a, c)
        reward = -F.logsigmoid(d)

        if self.use_posterior:  # TODO: add model.eval()
            next_s = s[1:]
            cur_a = a[:-1]
            pre_opt = c_1[:-1]
            target_opt = c[:-1]

            onehot_opt = F.one_hot(pre_opt.squeeze(-1), num_classes=self.dim_c + 1)
            if self.gru_include_action:
                gru_input = torch.cat([next_s, cur_a, onehot_opt], dim=-1)
            else:
                gru_input = torch.cat([next_s, onehot_opt], dim=-1)
            gru_input = gru_input.unsqueeze(1)  # batch_size is 1; no gradient info
            gru_output = self.posterior(gru_input)
            gru_logp_array = F.log_softmax(gru_output, dim=-1)
            gru_logp = gru_logp_array.gather(dim=-1, index=target_opt)

            gru_logp = torch.cat([gru_logp, torch.zeros((1, 1), dtype=torch.float32).to(gru_logp.device)],
                                 dim=0).detach().clone()
            reward = reward + self.alpha_2 * gru_logp  # note that the entropy term will be included in the PPO part later

        return reward

    def step(self, sample_scar, demo_scar, n_step=10):
        # Posterior training
        if self.use_posterior:
            print("Training the posterior......")
            for _ in range(self.gru_training_iters):
                for s, c, a, r in sample_scar:
                    next_s = s[1:]
                    cur_a = a[:-1]
                    pre_opt = c[:-2]
                    target_opt = c[1:-1]
                    onehot_opt = F.one_hot(pre_opt.squeeze(-1), num_classes=self.dim_c + 1)
                    if self.gru_include_action:
                        gru_input = torch.cat([next_s, cur_a, onehot_opt], dim=-1)
                    else:
                        gru_input = torch.cat([next_s, onehot_opt], dim=-1)
                    gru_input = gru_input.unsqueeze(1)  # batch_size is 1; no gradient info
                    gru_output = self.posterior(gru_input)  # (seq_len, dim_c)
                    # start training
                    gru_logp_array = F.log_softmax(gru_output, dim=-1)  # (seq_len, dim_c)
                    gru_logp = gru_logp_array.gather(dim=-1, index=target_opt)  # (seq_len, 1)
                    gru_loss = -torch.mean(gru_logp)

                    self.gru_optim.zero_grad()
                    gru_loss.backward()
                    self.gru_optim.step()

        # Discriminator training
        sp = torch.cat([s for s, c, a, r in sample_scar], dim=0)
        se = torch.cat([s for s, c, a, r in demo_scar], dim=0)
        c_1p = torch.cat([c[:-1] for s, c, a, r in sample_scar], dim=0)
        c_1e = torch.cat([c[:-1] for s, c, a, r in demo_scar], dim=0)
        cp = torch.cat([c[1:] for s, c, a, r in sample_scar], dim=0)
        ce = torch.cat([c[1:] for s, c, a, r in demo_scar], dim=0)
        ap = torch.cat([a for s, c, a, r in sample_scar], dim=0)
        ae = torch.cat([a for s, c, a, r in demo_scar], dim=0)
        tp = torch.ones(self.mini_bs, 1, dtype=torch.float32, device=self.device)
        te = torch.zeros(self.mini_bs, 1, dtype=torch.float32, device=self.device)

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
                for _ in range(3):
                    src = self.discriminator.get_unnormed_d(s_array, c_1array, a_array, c_array)
                    loss = F.binary_cross_entropy_with_logits(src, t_array)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()


    def convert_demo(self, demo_sa):
        with torch.no_grad(): # important
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                if not self.use_posterior:
                    c_array, _ = self.policy.viterbi_path(s_array, a_array)
                else:
                    if not self.use_posterior_sampling:
                        c_array, _ = self.policy.viterbi_path(s_array, a_array)
                    else: # of high variance, but very quick
                        print("Generating the option code sequence with the posterior......")
                        seq_len = int(s_array.size(0))
                        c_array = torch.zeros(s_array.size(0) + 1, 1, dtype=torch.long, device=self.device)
                        c_array[0] = self.dim_c
                        hidden = self.posterior.init_hidden()
                        for i in range(1, seq_len):
                            pre_opt = F.one_hot(c_array[i-1], num_classes=self.dim_c + 1)
                            next_s = s_array[i].unsqueeze(0)
                            cur_a = a_array[i-1].unsqueeze(0)
                            if self.gru_include_action:
                                gru_input = torch.cat([next_s, cur_a, pre_opt], dim=-1)
                            else:
                                gru_input = torch.cat([next_s, pre_opt], dim=-1)

                            gru_input = gru_input.unsqueeze(1)  # batch_size is 1; no gradient info
                            gru_output, hidden = self.posterior.forward_step(gru_input, hidden) # no grad info
                            gru_logp_array = F.log_softmax(gru_output, dim=-1)
                            # it's not appropriate to use argmax, since it will be greedy and can't guarantee the optimism
                            # use gumbel softmax to be in line with the high-level policy
                            opt = F.gumbel_softmax(gru_logp_array, hard=False).multinomial(1).long()  # (1, 1)
                            c_array[i] = opt
                        # given that we don't have S_T, we have to use the option_policy to sample the C_T
                        c_array[-1] = self.policy.sample_option(s_array[-1].unsqueeze(0), c_array[-2].unsqueeze(0))

                r_array = self.gail_reward(s_array, c_array[:-1], a_array, c_array[1:])
                out_sample.append((s_array, c_array, a_array, r_array))
                r_sum_avg += r_array.sum().item()
            r_sum_avg /= len(demo_sa)
        return out_sample, r_sum_avg

    def convert_sample(self, sample_scar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, c_array, a_array, r_real_array in sample_scar:
                r_fake_array = self.gail_reward(s_array, c_array[:-1], a_array, c_array[1:])
                out_sample.append((s_array, c_array, a_array, r_fake_array))
                r_sum_avg += r_real_array.sum().item()
            r_sum_avg /= len(sample_scar)
        return out_sample, r_sum_avg
