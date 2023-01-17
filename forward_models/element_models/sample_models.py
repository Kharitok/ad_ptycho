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




# ___________Sample models___________


class Sample_complex_TF(th.nn.Module):
    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif not (init_sample is None):
            self.sample = nn.Parameter(th.from_numpy(init_sample).cfloat())
        else:
            self.sample = nn.Parameter(th.ones(sample_size, dtype=th.complex64))

    def forward(self):
        return self.sample


class Sample_double_real_TF(th.nn.Module):
    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif not (init_sample is None):
            self.sample_real = nn.Parameter(th.from_numpy(init_sample.real).float())
            self.sample_imag = nn.Parameter(th.from_numpy(init_sample.imag).float())
        else:
            self.sample_real = nn.Parameter(th.ones(sample_size, dtype=th.float32))
            self.sample_imag = nn.Parameter(th.zeros(sample_size, dtype=th.float32))

    def forward(self):
        return th.complex(self.sample_real, self.sample_imag)


class Sample_refractive(th.nn.Module):
    def __init__(self, sample_size=None, init_sample=None):
        super().__init__()

        if (sample_size is None) and (init_sample is None):
            raise ValueError("Either sample_size or init_sample should be given")
        elif not (init_sample is None):
            self.sample = nn.Parameter(th.from_numpy(init_sample).cfloat())
        else:
            self.sample = nn.Parameter(th.zeros(sample_size, dtype=th.complex64))

    def forward(self):
        return th.exp(1j * self.sample)

    def get_transmission_and_pase(self):

        trans = th.exp(-1 * th.imag(self.sample))
        phase = th.real(self.sample)
        return (trans, phase)


class Sample_thickness(th.nn.Module):
    pass