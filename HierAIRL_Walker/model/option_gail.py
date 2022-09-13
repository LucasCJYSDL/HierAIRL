import torch
import torch.nn.functional as F
from .option_policy import OptionPolicy, Policy
from .option_discriminator import OptionDiscriminator, Discriminator
from utils.config import Config
from utils.model_util import clip_grad_norm_


class GAIL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(GAIL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.device = torch.device(config.device)
        self.mini_bs = config.mini_batch_size
        lr = config.optimizer_lr_discriminator

        self.discriminator = Discriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = Policy(config, dim_s=dim_s, dim_a=dim_a)

        self.optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr, weight_decay=3.e-5) # only update the disc and fix the policy

        self.to(self.device)

    def gail_reward(self, s, a):
        d = self.discriminator.get_unnormed_d(s, a)
        exp_d = torch.exp(d)
        sig_d = (exp_d / (exp_d + 1.0)).detach().clone()
        reward = sig_d
        return reward

    def step(self, sample_sar, demo_sar, n_step=10):
        sp = torch.cat([s for s, a, r in sample_sar], dim=0)
        se = torch.cat([s for s, a, r in demo_sar], dim=0)
        ap = torch.cat([a for s, a, r in sample_sar], dim=0)
        ae = torch.cat([a for s, a, r in demo_sar], dim=0)
        tp = torch.zeros(self.mini_bs, 1, dtype=torch.float32,
                        device=self.device)  # label for the generated state-action pairs
        te = torch.ones(self.mini_bs, 1, dtype=torch.float32,
                         device=self.device)  # label for the expert state-action pairs

        for _ in range(n_step):
            inds = torch.randperm(sp.size(0), device=self.device)
            for ind_p in inds.split(self.mini_bs):
                sp_b, ap_b, tp_b = sp[ind_p], ap[ind_p], tp[:ind_p.size(0)]
                ind_e = torch.randperm(se.size(0), device=self.device)[:ind_p.size(0)]
                se_b, ae_b, te_b = se[ind_e], ae[ind_e], te[:ind_p.size(0)]

                s_array = torch.cat((sp_b, se_b), dim=0)
                a_array = torch.cat((ap_b, ae_b), dim=0)
                t_array = torch.cat((tp_b, te_b), dim=0)
                for _ in range(3):
                    src = self.discriminator.get_unnormed_d(s_array, a_array)
                    loss = F.binary_cross_entropy_with_logits(src, t_array)
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

    def convert_demo(self, demo_sa):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                r_fake_array = self.gail_reward(s_array, a_array)
                out_sample.append((s_array, a_array, r_fake_array))
                r_sum_avg += r_fake_array.sum().item()
            r_sum_avg /= len(demo_sa)
        return out_sample, r_sum_avg

    def convert_sample(self, sample_sar):
        with torch.no_grad():
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array, r_real_array in sample_sar:
                r_fake_array = self.gail_reward(s_array, a_array)
                out_sample.append((s_array, a_array, r_fake_array))
                r_sum_avg += r_real_array.sum().item()
            r_sum_avg /= len(sample_sar)
        return out_sample, r_sum_avg


class OptionGAIL(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(OptionGAIL, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_c = config.dim_c
        self.with_c = config.use_c_in_discriminator
        self.mini_bs = config.mini_batch_size
        self.use_d_info_gail = config.use_d_info_gail
        self.device = torch.device(config.device)

        self.discriminator = OptionDiscriminator(config, dim_s=dim_s, dim_a=dim_a)
        self.policy = OptionPolicy(config, dim_s=self.dim_s, dim_a=self.dim_a)

        self.optim = torch.optim.Adam(self.discriminator.parameters(), weight_decay=1e-3) # 1e-3
        self.to(self.device)

    def original_gail_reward(self, s, c_1, a, c):
        d = self.discriminator.get_unnormed_d(s, c_1, a, c)
        reward = -F.logsigmoid(d)
        return reward

    def d_info_gail_reward(self, s, c_1, a, c):
        d = self.discriminator.get_unnormed_d(s, c_1, a, c)
        # la, lb, _, _, _ = self.policy.log_alpha_beta(s, a)
        # logpc = (la + lb).log_softmax(dim=-1).gather(dim=-1, index=c)
        reward = -F.logsigmoid(d)
        # 0.001 comes from the original implementation?
        reward += 0.001 * self.policy.log_prob_option(s, c_1, c) # entropy term is in PPO
        return reward

    def gail_reward(self, s, c_1, a, c):
        if not self.use_d_info_gail:
            return self.original_gail_reward(s, c_1, a, c)
        else:
            return self.d_info_gail_reward(s, c_1, a, c)
            # print("The reward for the DI-GAIL has not been implemented yet.")
            # raise NotImplementedError

    def step(self, sample_scar, demo_scar, n_step=10):
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
                for _ in range(3): # 3
                    src = self.discriminator.get_unnormed_d(s_array, c_1array, a_array, c_array)
                    loss = F.binary_cross_entropy_with_logits(src, t_array)
                    self.optim.zero_grad()
                    loss.backward()
                    # clip_grad_norm_(self.discriminator.parameters(), max_norm=20, norm_type=2)
                    self.optim.step()


    def convert_demo(self, demo_sa):
        with torch.no_grad(): # important
            out_sample = []
            r_sum_avg = 0.
            for s_array, a_array in demo_sa:
                if self.with_c:
                    c_array, _ = self.policy.viterbi_path(s_array, a_array)
                else:
                    c_array = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
                    # this is reasonable, since the c's are not used as part of input for DI-GAIL
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


