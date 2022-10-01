import math
import torch
from torch import nn
from utils.MHA_model_util import layer_init, range_tensor, DoeDecoderFFN, SkillPolicy, DoeSingleTransActionNet, DoeCriticNet
from utils.config import Config
import torch.nn.functional as F

class MHAOptionPolicy(nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(MHAOptionPolicy, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dmodel = config.dmodel
        self.dim_c = config.dim_c
        self.device = torch.device(config.device)
        self.log_clamp = config.log_clamp_policy

        # option embedding
        self.embed_option = nn.Embedding(config.dim_c, self.dmodel) # embedding matrix and the kernel matrix of the attention module
        nn.init.orthogonal(self.embed_option.weight) # initialization

        ## option policy:
        self.init_option_policy = DoeDecoderFFN(self.dim_s, hidden_units=(64, self.dim_c))  # used for the first step when the previous option is not available

        self.de_state_lc = layer_init(nn.Linear(self.dim_s, self.dmodel))
        self.de_state_norm = nn.LayerNorm(self.dmodel)
        self.de_logtis_lc = layer_init(nn.Linear(2 * self.dmodel, self.dim_c))
        self.doe = SkillPolicy(self.dmodel, config.mha_nhead, config.mha_nlayers, config.mha_nhid, config.dropout)

        # policy
        # self.act_concat_norm = nn.LayerNorm(self.dim_s + self.dmodel)
        self.act_doe = DoeSingleTransActionNet(self.dim_s + self.dmodel, self.dim_a, hidden_units=config.hidden_policy)

        self.to(self.device)

    def a_mean_logstd(self, st, ct=None):
        # ct: None or long(N x 1)
        # ct: None for all c, return (N x dim_c x dim_a); else return (N x dim_a)
        # s: N x dim_s, c: N x 1, c should always < dim_c

        if ct is not None:
            ct = ct.squeeze(-1) # (N, )
            ct_emb = self.embed_option(ct.unsqueeze(0)).detach().squeeze(0) # (N, dim_e)
            act_inp = torch.cat([st, ct_emb], dim=-1)
            # act_inp = self.act_concat_norm(act_inp) # TODO
            mean, log_std = self.act_doe(act_inp)

        else:
            # embed_all_idx: [bs, dim_c]
            bs = st.shape[0]
            embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1) # (bs, dim_c)
            embed_all = self.embed_option(embed_all_idx).detach() # (bs, dim_c, dim_e)
            st = st.unsqueeze(1).repeat(1, self.dim_c, 1) # (bs, dim_c, dim_s)
            act_inp = torch.cat([st, embed_all], dim=-1).view(-1, self.dim_s + self.dmodel)
            mean, log_std = self.act_doe(act_inp)
            mean = mean.view(bs, self.dim_c, self.dim_a)
            log_std = log_std.view(bs, self.dim_c, self.dim_a)

        return mean.clamp(-10, 10), log_std.clamp(self.log_clamp[0], self.log_clamp[1]) # TODO

    def switcher(self, s):
        # s: (N, dim_s)
        # output: (N x ct_1 x ct)
        # encoder inputs
        bs = s.shape[0]
        # embed_all_idx: [dim_c, bs]
        embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1).t()
        wt = self.embed_option(embed_all_idx) # (dim_c, bs, dim_e) # this is the attention kernel matrix

        # state input
        s_rep = s.unsqueeze(1).repeat(1, self.dim_c, 1).view(-1, self.dim_s) # (bs*dim_c, dim_s)
        s_hat = F.relu(self.de_state_lc(s_rep)) # (bs*dim_c, dim_e)
        s_hat = self.de_state_norm(s_hat) # (bs*dim_c, dim_e)
        # option input
        embed_all_idx = range_tensor(self.dim_c, self.device).repeat(bs, 1)  # (bs, dim_c)
        prev_options = embed_all_idx.view(-1, 1) # (bs*dim_c, 1)
        ct_1 = self.embed_option(prev_options.t()).detach()  # (1, bs*dim_c, dim_e)
        # concat
        opt_cat_1 = torch.cat([s_hat.unsqueeze(0), ct_1], dim=0) # (2, bs*dim_c, dim_e)
        rdt = self.doe(wt, opt_cat_1) # (2, bs*dim_c, dim_e)
        dt = torch.cat([rdt[0].squeeze(0), rdt[1].squeeze(0)], dim=-1) # (bs*dim_c, 2*dim_e)
        opt_logits = self.de_logtis_lc(dt) # (bs*dim_c, dim_c)
        opt_logits = opt_logits.view(bs, self.dim_c, self.dim_c)

        ## deal with the init option
        opt_logits_init = self.init_option_policy(s) # (bs, dim_c)
        opt_logits_init = opt_logits_init.unsqueeze(1) # (bs, 1, dim_c)

        logits = torch.cat([opt_logits, opt_logits_init], 1) # (bs, dim_c+1, dim_c), note that the first step option is dim_c

        return logits

    def log_trans(self, st, ct_1=None):
        # ct_1: long(N x 1) or None
        # ct_1: None: direct output p(ct|st, ct_1): a (N x ct_1 x ct) array where ct is log-normalized
        unnormed_pcs = self.switcher(st) # (bs, dim_c+1, dim_c)
        log_pcs = unnormed_pcs.log_softmax(dim=-1) # (bs, dim_c+1, dim_c)
        if ct_1 is None:
            return log_pcs
        else:
            return log_pcs.gather(dim=-2, index=ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)).squeeze(dim=-2) # (bs, dim_c)

    # TODO
    # def get_param(self, low_policy=True):
    #     parameter = []
    #     if low_policy:
    #         for name, para in self.named_parameters():
    #             if 'act_doe' in name:
    #                 parameter.append(para)
    #     else:
    #         for name, para in self.named_parameters():
    #             if 'act_doe' not in name:
    #                 print(name)
    #                 parameter.append(para)
    #
    #     return parameter

    def get_param(self, low_policy=True):
        return list(self.parameters())

    def log_prob_action(self, st, ct, at): # Gaussian Distribution, independence assumption on the dimensions of actions
        # if c is None, return (N x dim_c x 1), else return (N x 1)
        mean, logstd = self.a_mean_logstd(st, ct)
        if ct is None:
            at = at.view(-1, 1, self.dim_a)
        return (-((at - mean).square()) / (2 * (logstd * 2).exp()) - logstd - math.log(math.sqrt(2 * math.pi))).sum(dim=-1, keepdim=True)

    def log_prob_option(self, st, ct_1, ct):
        log_tr = self.log_trans(st, ct_1)
        return log_tr.gather(dim=-1, index=ct)

    def sample_action(self, st, ct, fixed=False):
        action_mean, action_log_std = self.a_mean_logstd(st, ct)
        if fixed:
            action = action_mean
        else:
            eps = torch.empty_like(action_mean).normal_()
            action = action_mean + action_log_std.exp() * eps
        return action

    def sample_option(self, st, ct_1, fixed=False, tau=1.0):
        log_tr = self.log_trans(st, ct_1)
        if fixed:
            return log_tr.argmax(dim=-1, keepdim=True)
        else:
            # print(F.gumbel_softmax(log_tr, hard=False)) # (N, c_dim)
            # Note that the sampling result does not contain gradients.
            return F.gumbel_softmax(log_tr, hard=False, tau=tau).multinomial(1).long() # (N, 1) it's a surprise that option-gail has implemented this

    def policy_entropy(self, st, ct):
        _, log_std = self.a_mean_logstd(st, ct)
        entropy = 0.5 + 0.5 * math.log(2 * math.pi) + log_std
        return entropy.sum(dim=-1, keepdim=True) # (bs, 1)

    def option_entropy(self, st, ct_1):
        log_tr = self.log_trans(st, ct_1)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return entropy # (bs, 1)

    def policy_log_prob_entropy(self, st, ct, at):
        mean, logstd = self.a_mean_logstd(st, ct)
        log_prob = (-(at - mean).pow(2) / (2 * (logstd * 2).exp()) - logstd - 0.5 * math.log(2 * math.pi)).sum(dim=-1, keepdim=True)
        entropy = (0.5 + 0.5 * math.log(2 * math.pi) + logstd).sum(dim=-1, keepdim=True)
        return log_prob, entropy

    def option_log_prob_entropy(self, st, ct_1, ct):
        # c1 can be dim_c, c2 should always < dim_c
        log_tr = self.log_trans(st, ct_1)
        log_opt = log_tr.gather(dim=-1, index=ct)
        entropy = -(log_tr * log_tr.exp()).sum(dim=-1, keepdim=True)
        return log_opt, entropy

    def viterbi_path(self, s_array, a_array):
        # exactly follow the Eq. (8)-(10) in the paper
        with torch.no_grad():
            log_pis = self.log_prob_action(s_array, None, a_array).view(-1, 1, self.dim_c)  # demo_len x 1 x ct
            log_trs = self.log_trans(s_array, None)  # demo_len x (ct_1+1) x ct
            log_prob = log_trs[:, :-1] + log_pis # demo_len x ct_1 x ct
            log_prob0 = log_trs[0, -1] + log_pis[0, 0] # (ct, )
            # forward
            max_path = torch.empty(s_array.size(0), self.dim_c, dtype=torch.long, device=self.device)
            accumulate_logp = log_prob0 # (ct, )
            max_path[0] = self.dim_c
            for i in range(1, s_array.size(0)):
                accumulate_logp, max_path[i, :] = (accumulate_logp.unsqueeze(dim=-1) + log_prob[i]).max(dim=-2)
            # backward
            c_array = torch.zeros(s_array.size(0)+1, 1, dtype=torch.long, device=self.device)
            log_prob_traj, c_array[-1] = accumulate_logp.max(dim=-1)
            for i in range(s_array.size(0), 0, -1):
                c_array[i-1] = max_path[i-1][c_array[i]]
        return c_array.detach(), log_prob_traj.detach()


class MHAOptionCritic(torch.nn.Module):
    def __init__(self, config, dim_s, dim_c):
        super(MHAOptionCritic, self).__init__()
        self.dim_s = dim_s
        self.dim_c = dim_c
        self.dmodel = config.dmodel
        self.device = torch.device(config.device)

        critic_dim = self.dim_s + self.dmodel
        self.q_concat_norm = nn.LayerNorm(critic_dim)
        self.q_o_st = DoeCriticNet(critic_dim, self.dim_c, config.hidden_critic)

        self.to(self.device)

    def get_param(self):
        return list(self.parameters())

    def get_value(self, option_emb, st, ct, return_all=True): # TODO: try the original get_value
        # st: (bs, dim_s), option_emb: coming from the current policy, ct: (bs, 1)
        # this logic is weird, but the orginal code of SA uses this and get good performance
        # the best way to solve this may be to train separate critics for the low- and high-level policy
        assert ct is not None
        ct = ct.squeeze(-1)  # (N, )
        ct_emb = option_emb(ct.unsqueeze(0)).detach().squeeze(0)  # (N, dim_e)
        v_inp = torch.cat([st, ct_emb], dim=-1) # (N, dim_s + dim_e)
        v_inp = self.q_concat_norm(v_inp)
        q_o_st = self.q_o_st(v_inp) # (N, dim_c)

        if return_all:
            return q_o_st
        else:
            return q_o_st.gather(dim=-1, index=ct.unsqueeze(-1))