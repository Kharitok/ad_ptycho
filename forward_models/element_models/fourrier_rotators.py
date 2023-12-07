"""
Fourrier rotators  for AD ptychography
"""
import torch.nn.functional as F
import torch.nn as nn
import torch as th

import torch as th
import torch.nn as nn
import numpy as np
import torch.fft as th_fft

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader
# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True

def th_ff_sampling(field):
    """FFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.fft2(th_fft.ifftshift(field, dim=(-1, -2)), norm="ortho"), dim=(-1, -2)
    )


def th_iff_sampling(field):
    """IFFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.ifft2(th_fft.ifftshift(field, dim=(-1, -2)), norm="ortho"), dim=(-1, -2)
    )



class Bulk_fft_upsampler_Flex(th.nn.Module):
    """Upsamples probe or sample to match the propagation conditions"""
    def __init__(self):
        super().__init__()
        

    def forward(self, X):
        """Performs forward propagation"""
        return  th_iff(th.nn.functional.pad(th_ff_sampling(X),(512,512,512,512)))*4096

    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff_sampling(th_ff_sampling(X)[512:-512,512:-512])





class Bulk_fft_upsampler(th.nn.Module):
    """Upsamples probe or sample to match the propagation conditions"""
    def __init__(self):
        super().__init__()
        

    def forward(self, X):
        """Performs forward propagation"""
        return  th_iff(th.nn.functional.pad(th_ff_sampling(X),(512,512,512,512)))*4096

    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff_sampling(th_ff_sampling(X)[512:-512,512:-512])


