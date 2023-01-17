"""
Contains models required for the AD-based ptychography
"""
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import torch.fft as th_fft
from torch.utils.data import Dataset, DataLoader
# from propagators import grid

th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
th.backends.cudnn.benchmark = True


# ___________Probe models___________


class Probe_complex_shot_to_shot_constant(th.nn.Module):
    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

    def forward(self, scan_numbers):
        return self.probe[None, ...]


class Probe_double_real_shot_to_shot_constant(th.nn.Module):
    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe_real = nn.Parameter(
                (th.from_numpy(init_probe.real).float())[None, :, :]
            )
            self.probe_imag = nn.Parameter(
                (th.from_numpy(init_probe.imag).float())[None, :, :]
            )
        else:
            self.probe_real = nn.Parameter((th.from_numpy(init_probe.real).float()))
            self.probe_imag = nn.Parameter((th.from_numpy(init_probe.imag).float()))

    def forward(self, scan_numbers):
        return th.complex(self.probe_real, self.probe_imag)[None, ...]


class Probe_complex_shot_to_shot_variable(th.nn.Module):
    def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        if not (modal_weights is None):
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif not (number_of_positions is None):
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        return self.probe[None, :, :] * self.modal_weights[scan_numbers, :, None, None]


class Probe_double_real_shot_to_shot_variable(th.nn.Module):
    def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe_real = nn.Parameter(
                (th.from_numpy(init_probe.real).float())[None, :, :]
            )
            self.probe_imag = nn.Parameter(
                (th.from_numpy(init_probe.imag).float())[None, :, :]
            )
        else:
            self.probe_real = nn.Parameter((th.from_numpy(init_probe.real).float()))
            self.probe_imag = nn.Parameter((th.from_numpy(init_probe.imag).float()))

        if not (modal_weights is None):
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif not (number_of_positions is None):
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        return (
            th.complex(self.probe_real, self.probe_imag)[None, :, :]
            * self.modal_weights[scan_numbers, :, None, None]
        )

