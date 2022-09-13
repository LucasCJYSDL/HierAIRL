import torch
import numpy as np


class StateFilter(object):
    def __init__(self, enable=True):
        self.enable = enable
        self.mean = None
        self.var = None
        self.n_step = 0
        self.clamp = (-5., 5.)

    def __call__(self, x, fixed=False):
        if not self.enable:
            return x

        if self.mean is None or self.n_step < 1:
            if fixed:
                return x.clip(*self.clamp) if isinstance(x, np.ndarray) else x.clamp(*self.clamp)
            else:
                self.mean = x.copy()
                self.var = np.zeros_like(self.mean)
                self.n_step = 1
                return np.zeros_like(x) if isinstance(x, np.ndarray) else torch.zeros_like(x)
        else:
            is_tensor = isinstance(x, torch.Tensor)
            device = None
            size = None
            if is_tensor:
                device = x.device
                size = x.size()
                x = x.squeeze().cpu().numpy()
            if not fixed:
                oldM = self.mean
                self.n_step = self.n_step + 1
                self.mean = oldM + (x - oldM) / self.n_step
                self.var = (self.var * ((self.n_step - 2) / (self.n_step - 1)) +
                            (x - oldM) * (x - self.mean) / (self.n_step - 1))
            fx = ((x - self.mean) / (np.sqrt(self.var) + 1.e-8)).clip(*self.clamp)
            return torch.as_tensor(fx, dtype=torch.float32, device=device).resize_(*size) if is_tensor else fx

    def state_dict(self):
        return {
            "mean": self.mean,
            "var": self.var,
            "n_step": self.n_step
        }

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.var = state_dict["var"]
        self.n_step = state_dict["n_step"]
