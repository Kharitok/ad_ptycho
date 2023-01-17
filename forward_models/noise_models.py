"""
Contains noise models for ad-based ptychography
"""
# import torch.nn.functional as F
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


class Additive_Gaussian_noise(th.nn.Module):
    "Tries to estimate parameters of the additive gaussian-alike noise"

    def __init__(self, detector_size, x_init, y_init, sig_x_init, sig_y_init, max_init):

        super().__init__()

        self.init_parameters = th.tensor(
            [x_init, y_init, sig_x_init, sig_y_init, max_init]
        )
        self.gaussian_parameters = nn.Parameter(self.init_parameters.float())

        Xn, Yn = th.range(0, detector_size - 1), th.range(0, detector_size - 1)
        self.xx_, self.yy_ = Xn[None, :], Yn[:, None]
        self.register_buffer("xx", self.xx_.data)
        self.register_buffer("yy", self.yy_.data)

    def get_gaussian(self):
        return self.gaussian_parameters[4] * th.exp(
            -1
            * (
                (
                    (self.xx - self.gaussian_parameters[0]) ** 2
                    / (2 * self.gaussian_parameters[2] ** 2)
                )
                + (
                    (self.yy - self.gaussian_parameters[1]) ** 2
                    / (2 * self.gaussian_parameters[3] ** 2)
                )
            )
        )
