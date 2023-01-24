"""
Contains noise models for AD-based ptychography
"""
# import torch.nn.functional as F
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader
# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


class AdditiveGaussianNoise(th.nn.Module):
    """Parametriezed 2D gaussian noise generator, for use in the ptychography reconstructions
    to fit the background noise"""

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
        """Returns estimated intensity"""
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


class AdditiveGaussianNoiseVariable(th.nn.Module):
    """Parametriezed 2D gaussian noise generator, for use in the ptychography reconstructions
    to fit the background noise assuming shot-to-shot changing gaussian noise parameters
    """

    def __init__(
        self,
        detector_size,
        x_init,
        y_init,
        sig_x_init,
        sig_y_init,
        max_init,
        num_positions,
    ):
        super().__init__()

        self.init_parameters = (
            th.tensor([x_init, y_init, sig_x_init, sig_y_init, max_init])
            .unsqueeze(1)
            .repeat(1, num_positions)
        )
        self.gaussian_parameters = nn.Parameter(self.init_parameters.float())

        Xn, Yn = th.range(0, detector_size - 1), th.range(0, detector_size - 1)
        self.xx_, self.yy_ = Xn[None, :], Yn[:, None]
        self.register_buffer("xx", self.xx_.data)
        self.register_buffer("yy", self.yy_.data)

    def get_gaussian(self, pos_indx):
        """Returns estimated intensity"""
        gaussian_params_current = self.gaussian_parameters[..., pos_indx]
        new_dim = len(pos_indx)
        return (
            gaussian_params_current[4]
            * th.exp(
                -1
                * (
                    (
                        (
                            self.xx.unsqueeze(2).expand(-1, -1, new_dim)
                            - gaussian_params_current[0]
                        )
                        ** 2
                        / (2 * gaussian_params_current[2] ** 2)
                    )
                    + (
                        (
                            self.yy.unsqueeze(2).expand(-1, -1, new_dim)
                            - gaussian_params_current[1]
                        )
                        ** 2
                        / (2 * gaussian_params_current[3] ** 2)
                    )
                )
            )
        ).permute(2, 0, 1)
