from torch.fft import (
    fftshift as fftshift,
    fftn as fftn_t,
    ifftn as ifftn_t,
    ifftshift as ifftshift,
)
import torch.fft as th_fft
import torch as th
import numpy as np


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


def th_ff_sampling_3d(field):
    """FFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.fftn(th_fft.ifftshift(field, dim=(-1, -2, -3)), norm="ortho"),
        dim=(-1, -2, -3),
    )


def th_iff_sampling_3d(field):
    """IFFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.ifftn(th_fft.ifftshift(field, dim=(-1, -2, -3)), norm="ortho"),
        dim=(-1, -2, -3),
    )


def Pad(X, padding):
    return th.nn.functional.pad(X, padding)


def get_pad_size_for_bulk_resampling(
    shape: list[int], sampling_init: list[float], sampling_desired: list[float]
) -> th.Tensor:
    """
    Calculates the padding size required for bulk resampling of data.

    Args:
        shape list[int]: The spatial dimensions of the input data.
        sampling_init (list[float]): A list of initial sampling rates for each spatial dimension.
        sampling_desired (list[float]): A list of desired sampling rates for each spatial dimension.

    Returns:
        torch.Tensor: Padding required for bulk resampling
    """
    sampling_init = th.tensor(sampling_init[::-1])
    sampling_desired = th.tensor(sampling_desired[::-1])
    shape_t = th.tensor(shape[::-1])

    return (
        ((-1 / 2) * (shape_t * (1 - sampling_init / sampling_desired)))
        .int()
        .repeat_interleave(2)
    )


def Bulk_Resample_2d(X: th.Tensor, padding: th.Tensor) -> th.Tensor:
    """
    Resamples tensor along last two axis

    Args:
        X th.Tensor: Tensor to resample, with resampling performed for last two axis
        padding th.Tensor: a result of get_pad_size_for_bulk_resampling to change the sampling
    Returns:
        torch.Tensor: Resampled tensor
    """
    return Pad(
        th_iff_sampling(Pad(th_ff_sampling(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )


def Bulk_Resample_3d(X: th.Tensor, padding: th.Tensor) -> th.Tensor:
    """
    Resamples tensor along last three axis

    Args:
        X th.Tensor: Tensor to resample, with resampling performed for last three axis
        padding th.Tensor: a result of get_pad_size_for_bulk_resampling to change the sampling
    Returns:
        torch.Tensor: Resampled tensor
    """
    return Pad(
        th_iff_sampling_3d(Pad(th_ff_sampling_3d(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )
