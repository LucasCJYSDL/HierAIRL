import torch
from utils.model_util import make_module, make_module_list, make_activation
from utils.config import Config


# This file should be included by option_gail.py/option_airl.py and never be used otherwise


class Discriminator(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(Discriminator, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.device = torch.device(config.device)
        n_hidden_d = config.hidden_discriminator
        activation = make_activation(config.activation)

        self.discriminator = make_module(self.dim_s + self.dim_a, 1, n_hidden_d, activation)

        self.to(self.device)

    def get_unnormed_d(self, s, a):
        d = self.discriminator(torch.cat((s, a), dim=-1))
        return d

    def gradient_penalty(self, s, a, lam=10.):
        sa = torch.cat((s, a), dim=-1).requires_grad_()
        d = self.discriminator(sa)

        gradients = torch.autograd.grad(outputs=d, inputs=sa, grad_outputs=torch.ones_like(d),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam
        return gradient_penalty


class OptionDiscriminator(torch.nn.Module):
    def __init__(self, config: Config, dim_s=2, dim_a=2):
        super(OptionDiscriminator, self).__init__()
        self.dim_a = dim_a
        self.dim_s = dim_s
        self.dim_c = config.dim_c
        self.with_c = config.use_c_in_discriminator
        self.is_shared = config.shared_discriminator
        self.device = torch.device(config.device)
        n_hidden_d = config.hidden_discriminator
        activation = make_activation(config.activation)
        if not self.is_shared and self.with_c: # Too many networks
            self.discriminator = make_module_list(self.dim_s + self.dim_a, 1, n_hidden_d, (self.dim_c+1) * self.dim_c, activation)
        else:
            self.discriminator = make_module(self.dim_s + self.dim_a, ((self.dim_c+1) * self.dim_c) if self.with_c else 1, n_hidden_d, activation)

        self.to(self.device)

    def get_unnormed_d(self, st, ct_1, at, ct):
        s_a = torch.cat((st, at), dim=-1)
        if not self.is_shared and self.with_c:
            d = torch.cat([m(s_a) for m in self.discriminator], dim=-1) # [N, (dim_c+1) * dim_c]
        else:
            d = self.discriminator(s_a)
        if self.with_c:
            d = d.view(-1, self.dim_c+1, self.dim_c)
            ct_1 = ct_1.view(-1, 1, 1).expand(-1, 1, self.dim_c)
            d = d.gather(dim=-2, index=ct_1).squeeze(dim=-2).gather(dim=-1, index=ct) # ct should be [N, 1]
        return d

    def gradient_penalty(self, s, a, ct_1, ct, lam=10.):
        assert not self.is_shared and self.with_c
        sa = torch.cat((s, a), dim=-1).requires_grad_()
        # d = self.get_unnormed_d(s, ct_1, a, ct)
        d = torch.cat([m(sa) for m in self.discriminator], dim=-1)
        gradients = torch.autograd.grad(outputs=d, inputs=sa, grad_outputs=torch.ones_like(d),
                                        create_graph=True, retain_graph=True, only_inputs=True, allow_unused=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lam
        return gradient_penalty




