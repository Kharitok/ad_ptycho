"""
Contains propagators required for the AD-based ptychography
"""
import torch.nn.functional as F
import torch.nn as nn
import torch as th
import torch.fft as th_fft
from torch.utils.data import Dataset, DataLoader

th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
th.backends.cudnn.benchmark = True

# ___________Fourrier transform staff___________


def th_ff(a):
    """FFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.fft2(th_fft.ifftshift(a, dim=(-1, -2)), norm="backward"), dim=(-1, -2)
    )


def th_iff(a):
    """IFFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.ifft2(th_fft.ifftshift(a, dim=(-1, -2)), norm="backward"), dim=(-1, -2)
    )


# ___________Misc functions for propagators construction___________


def grid(
    pixel_size,
    pixel_num,
):
    """
    return grid of the field as a meshgrid
    """
    dx, dy = pixel_size, pixel_size
    (Nx,) = pixel_num, pixel_num

    grid_ = th.arange(0, Nx, 1) - Nx // 2

    Xn = grid_ * dx
    Yn = grid_ * dy

    xx, yy = Xn[None, :], Yn[:, None]

    return (xx, yy)


def freq_grid(pixel_size, pixel_num):
    """
    Returns Fourrier frequencies grid as a meshgrid
    """

    fx = th.fft.ifftshift(th.fft.fftfreq(n=pixel_num, d=pixel_size))
    fxx, fyy = fx[None, :], fx[:, None]

    return (fxx, fyy)


def fourrier_scaled_grid(
    lm,
    z,
    pixel_size,
    pixel_num,
):
    """
    Returns coordinate grid in the target plane acourding to the fourrier scaling
    dx2 = lm*z/(N*dx)
    """
    z = th.tensor(z)

    grid2 = th.arange(0, pixel_num, 1) - pixel_num // 2

    dx2 = (lm * th.abs(z) / (pixel_num * pixel_size)) * grid2
    dy2 = dx2

    x2, y2 = dx2[None, :], dy2[:, None]
    return (x2, y2)


# ___________Propagators___________


class Propagator_Fresnel_single_transform_flux_preserving(th.nn.Module):
    """
    Fresnel Propagator with complex torch
    """

    def __init__(
        self,
        pixel_size,
        pixel_num,
        wavelength,
        z,
    ):
        super().__init__()

        lm = wavelength
        k = 2 * th.pi / lm

        x1, y1 = grid(pixel_size, pixel_num)

        x2, y2 = fourrier_scaled_grid(lm, z, pixel_size, pixel_num)

        mul1 = th.exp((1j * k / (2 * z)) * (x2**2 + y2**2))
        mul2 = th.exp((1j * k / (2 * z)) * (x1**2 + y1**2))

        mul1_inv = th.exp((1j * k / (2 * (-z)) * (x1**2 + y1**2)))
        mul2_inv = th.exp((1j * k / (2 * (-z)) * (x2**2 + y2**2)))

        self.register_buffer("mul1", mul1.cfloat())
        self.register_buffer("mul2", mul2.cfloat())

        self.register_buffer("mul1_inv", mul1_inv.cfloat())
        self.register_buffer("mul2_inv", mul2_inv.cfloat())

        self.num = pixel_num

    def forward(self, X):
        return self.mul1 * th_ff(X * self.mul2) / self.num

    def inverse(self, X):
        return self.mul1_inv * th_iff(X * self.mul2_inv) * self.num


class Propagator_Fraunh_intensity_flux_preserving(th.nn.Module):
    """
    Fresnel Propagator with complex torch
    """

    def __init__(
        self,
        pixel_num,
    ):

        super().__init__()
        self.num = pixel_num

    def forward(self, X):
        return th_ff(X) / self.num

    def inverse(self, X):
        return th_iff(X) * self.num
