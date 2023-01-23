"""
Constains probe models for ad based ptychography
"""
# import torch.nn.functional as Fw
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# from propagators import grid

th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
th.backends.cudnn.benchmark = True


# ___________Probe models___________


class ProbeComplexShotToShotConstant(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with complex tensor.
    """

    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return self.probe[None, ...]


class ProbeDoubleRealShotToShotConstant(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with two real tensors.
    """

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
        """Returns probe function at scan_numbers positions"""
        return th.complex(self.probe_real, self.probe_imag)[None, ...]


class ProbeComplexShotToShotVariable(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe with  shot to shot unique
    modal weights. Implemented with complex tensor.
    """

    def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        if modal_weights is not None:
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif number_of_positions is not None:
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return self.probe[None, :, :] * self.modal_weights[scan_numbers, :, None, None]


class ProbeDoubleRealShotToShotVariable(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe with  shot to shot unique
    modal weights. Implemented with two real tensors.
    """

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

        if modal_weights is not None:
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif number_of_positions is not None:
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return (
            th.complex(self.probe_real, self.probe_imag)[None, :, :]
            * self.modal_weights[scan_numbers, :, None, None]
        )
