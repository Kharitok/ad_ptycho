import torch.fft as th_fft
from torch.fft import (
    fftshift as fftshift_t,
    ifftshift as ifftshift_t,
    fftn as fftn_t,
    ifftn as ifftn_t,
    fftfreq as fftfreq_t,
)
import torch as th
import numpy as np

import matplotlib.pyplot as plt

import ad_ptycho.Plotting_utils as plot_

from typing import List

# FFt routines for future convenience


def fft1d_t(X: th.Tensor, n: int) -> th.Tensor:
    """FFT on a tensor using torch.fft"""
    return fftshift_t(fftn_t(ifftshift_t(X, dim=[n]), dim=[n], norm="ortho"), dim=[n])


def ifft1d(X: th.Tensor, n: int) -> th.Tensor:
    """IFFT on a tensor using torch.fft"""
    return fftshift_t(ifftn_t(ifftshift_t(X, dim=[n]), dim=[n], norm="ortho"), dim=[n])


def fftnd_t(X: th.Tensor, n: int) -> th.Tensor:
    """1d FFT on a tensor using torch.fft"""
    return fftshift_t(fftn_t(ifftshift_t(X, dim=n), dim=n, norm="ortho"), dim=n)


def ifftnd_t(X, n):
    """1d FFT on a tensor using torch.fft"""
    return fftshift_t(ifftn_t(ifftshift_t(X, dim=n), dim=n, norm="ortho"), dim=n)


def pad(X, padding):
    return th.nn.functional.pad(X, padding)

### Shifts ####
def shift_3d_fourrier(X:th.tensor,shifts : th.tensor) -> th.tensor:
    """
    Shifts 3d Tensor utilizing Fourrier shift theorem
    Args:
    X (th.tensor): The input tensor.
    shifts (th.tensor): The shifts to apply in each dimension.
    Returns:
    th.tensor: The result of the 3D Fourier shift.
    """
    freq_z  = fftshift_t(fftfreq_t(X.shape[-3], 1),dim=(0))
    freq_x  = fftshift_t(fftfreq_t(X.shape[-2], 1),dim=(0))
    freq_y  = fftshift_t(fftfreq_t(X.shape[-1], 1),dim=(0))
    shift_exp = th.exp(-2j* th.pi * (freq_z[:,None,None]*shifts[0]+freq_x[None,:,None]*shifts[1]+freq_y[None,None,:]*shifts[2]))
    return ifftnd_t(fftnd_t(X,(0,1,2))*shift_exp,(0,1,2))


def shift_3d_fourrier_reduced(X:th.tensor,shifts : th.tensor) -> th.tensor:
    """
    Shifts 3d Tensor utilizing Fourrier shift theorem
    Args:
    X (th.tensor): The input tensor.
    shifts (th.tensor): The shifts to apply in each dimension.
    Returns:
    th.tensor: The result of the 3D Fourier shift.
    """
    freq_z  = fftfreq_t(X.shape[-3], 1)
    freq_x  = fftfreq_t(X.shape[-2], 1)
    freq_y  = fftfreq_t(X.shape[-1], 1)
    shift_exp = th.exp(-2j* th.pi * (freq_z[:,None,None]*shifts[0]+freq_x[None,:,None]*shifts[1]+freq_y[None,None,:]*shifts[2]))
    return ifftn_t(fftn_t(X,dim=(0,1,2))*shift_exp,dim=(0,1,2))


### Bulk resampling routines ###


def get_pad_size_for_bulk_resampling(
    shape: List[int], sampling_init: List[float], sampling_desired: List[float]
) -> th.Tensor:
    """
    Calculates the padding size required for bulk resampling of data.

    Args:
        shape list[int]: The spatial dimensions of the input data.
        sampling_init (list[float]): A list of initial sampling rates for each spatial dimension.
        sampling_desired (list[float]): A list of desired sampling rates for each spatial dimension.

    Returns:
        torch.Tensor: padding required for bulk resampling
    """
    sampling_init = th.tensor(sampling_init[::-1])
    sampling_desired = th.tensor(sampling_desired[::-1])
    shape_t = th.tensor(shape[::-1][: len(sampling_init)])

    return (
        ((-1 / 2) * (shape_t * (1 - sampling_init / sampling_desired)))
        .int()
        .repeat_interleave(2)
    )


def bulk_resample_nd(X: th.Tensor, padding: th.Tensor) -> th.Tensor:
    """
    Resamples tensor along last N axis accourding to the padding length

    Args:
        X th.Tensor: Tensor to resample, with resampling performed for last N axis
        padding th.Tensor: a result of get_pad_size_for_bulk_resampling to change the sampling
    Returns:
        torch.Tensor: Resampled tensor
    """
    ft_axis = tuple(-1 * (i + 1) for i in range(padding.shape[0] // 2))
    return pad(
        ifftnd_t(pad(fftnd_t(X, ft_axis), tuple(padding.tolist())), ft_axis),
        tuple((-1 * padding).tolist()),
    )


### EXAMPLE ###

# test_cube = th.zeros(20, 20, 20, dtype=th.cfloat)
# test_cube[..., 5:15, 5:15, 5:15] = 1 - 1j
# test_cubes = th.unsqueeze(test_cube, 0).expand(10, -1, -1, -1)


# plot_.show_3d(test_cube)
# padding = get_pad_size_for_bulk_resampling(test_cube.shape,[1,1],[0.5,2])
# padded_cube = bulk_resample_nd(test_cube,padding)
# plot_.show_3d(padded_cube)

### END ###

### Shear routines ###


# 2D
def shear_2D(
    X: th.Tensor,
    magnitude: float,
    axis: int = -1,
) -> th.Tensor:
    """
    Shear 2D tensor along  axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """
    if axis == -1:
        co_axis = -2
        axis_slice = (None, ...)
        co_axis_slice = (..., None)
    elif axis == -2:
        co_axis = -1
        axis_slice = (..., None)
        co_axis_slice = (None, ...)

    freq_x = fftshift_t(fftfreq_t(X.shape[axis], 1), dim=(0))[axis_slice]
    pix_x = (
        th.linspace(-1, 1, X.shape[co_axis]) * (X.shape[co_axis] // 2) * magnitude
    )[co_axis_slice]
    shift_exp = th.exp(-2j * th.pi * freq_x * pix_x)
    return ifftnd_t(fftnd_t(X, (axis)) * (shift_exp), (axis))


### EXAMPLE ###
# test_cube = th.zeros(20, 20, 20, dtype=th.cfloat)
# test_cube[..., 5:15, 5:15, 5:15] = 1 - 1j
# plt.figure()
# plt.imshow(th.abs(shear_2D(test_cube[5,:,:],magnitude =1,axis = -2)),origin = 'lower')

### END ###


# 3D

# routines for axis calculations


def before_2(axis: int) -> int:
    return {-1: 1, -2: 0, -3: 2}.get(axis)


def after_2(axis: int) -> int:
    return {-1: 1, -2: 2, -3: 0}.get(axis)


def ax_2(axis: int) -> int:
    return {-1: -2, -2: -3, -3: -1}.get(axis)


def ax_3(axis: int) -> int:
    return {-1: -3, -2: -1, -3: -2}.get(axis)


def before_3(axis: int) -> int:
    return {-1: 0, -2: 0, -3: 1}.get(axis)


def after_3(axis: int) -> int:
    return {-1: 2, -2: 0, -3: 1}.get(axis)


def shear_3D(X: th.Tensor, axis: int, mag2: float, mag3: float) -> th.Tensor:
    """
    Shears 3D tensor along the axis
    X tensor to shear
    axis int in [-3,-2,-1] - axis to shear along
    mag2 float in [-1..1]  - magnitude of shear wrt to axis-1
    mag3 float in [-1..1]  - magnitude of shear wrt to axis-2
    """

    freq_0 = fftshift_t(fftfreq_t(X.shape[axis], 1), dim=(0))[
        (None,) * (3 + axis) + (...,) + (None,) * abs(1 + axis)
    ]  # [None,None,:]

    pix_2 = (
        th.linspace(-1, 1, X.shape[ax_2(axis)]) * (X.shape[ax_2(axis)] // 2) * mag2
    )[(None,) * (before_2(axis)) + (...,) + (None,) * (after_2(axis))]
    pix_3 = (
        th.linspace(-1, 1, X.shape[ax_3(axis)]) * (X.shape[ax_3(axis)] // 2) * mag3
    )[(None,) * (before_3(axis)) + (...,) + (None,) * (after_3(axis))]

    shift_exp = th.exp(-2j * th.pi * freq_0 * (pix_2 + pix_3))

    return ifftnd_t(fftnd_t(X, (axis)) * shift_exp, (axis))


### EXAMPLE ###
# test_cube = th.zeros(20, 20, 20, dtype=th.cfloat)
# test_cube[..., 5:15, 5:15, 5:15] = 1 - 1j


# plot_.show_3d(test_cube)
# plot_.show_3d(shear_3D(test_cube,axis = -1,mag2 = 1,mag3=-0.5))
# plot_.show_3d(shear_3D(test_cube,axis = -2,mag2 = 1,mag3=-0.5))
# plot_.show_3d(shear_3D(test_cube,axis = -3,mag2 = 1,mag3=-0.5))

### END ###

### Rotation routines ###

## 2D rotation


def rotate_2d(X: th.Tensor, theta: float) -> th.Tensor:
    """
    Rotates 2D tensor by the angle of theta radians
    """
    theta *= -1
    a = np.tan(theta / 2)
    b = -np.sin(theta)

    return shear_2D(
        shear_2D(shear_2D(X, axis=-1, magnitude=a), axis=-2, magnitude=b),
        axis=-1,
        magnitude=a,
    )


### EXAMPLE ###
# test_cube = th.zeros(20, 20, 20, dtype=th.cfloat)
# test_cube[..., 5:15, 5:15, 5:15] = 1 - 1j
# plt.figure()
# plt.imshow(th.abs(rotate_2d(test_cube[5,:,:],np.radians(10))),origin = 'lower')
### END ###


# 3D rotation


def rotate_3D_m1(X: th.Tensor, theta: float) -> th.Tensor:
    """
    Apply a 3D rotation to an input tensor around -1 axis.
    Args:
        X (Tensor): The input tensor to be rotated.
        theta (float): The rotation angle in radians.
    Returns:
        Tensor: The rotated tensor.
    """

    axis = -1

    theta *= -1

    co_axis_1 = -3
    co_axis_1_slice = (
        ...,
        None,
        None,
    )
    co_axis_2 = -2
    co_axis_2_slice = (None, ..., None)
    a = np.tan(theta / 2)
    b = -np.sin(theta)

    # phase_prefactor_a
    # phase_prefactor_b

    freq_c1 = fftshift_t(fftfreq_t(X.shape[co_axis_1], 1), dim=(0))[co_axis_1_slice]
    freq_c2 = fftshift_t(fftfreq_t(X.shape[co_axis_2], 1), dim=(0))[co_axis_2_slice]
    pix_1 = (th.linspace(-1, 1, X.shape[co_axis_1]) * (X.shape[co_axis_1] // 2) * b)[
        co_axis_1_slice
    ]
    pix_2 = (th.linspace(-1, 1, X.shape[co_axis_2]) * (X.shape[co_axis_2] // 2) * a)[
        co_axis_2_slice
    ]

    shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
    shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)

    return ifftnd_t(
        fftnd_t(
            ifftnd_t(
                fftnd_t(
                    ifftnd_t(fftnd_t(X, (co_axis_1)) * (shift_exp_c1), (co_axis_1)),
                    (co_axis_2),
                )
                * (shift_exp_c2),
                (co_axis_2),
            ),
            (co_axis_1),
        )
        * (shift_exp_c1),
        (co_axis_1),
    )


def rotate_3D_m2(X: th.Tensor, theta: float) -> th.Tensor:
    """
    Apply a 3D rotation to an input tensor around -2 axis.
    Args:
        X (Tensor): The input tensor to be rotated.
        theta (float): The rotation angle in radians.
    Returns:
        Tensor: The rotated tensor.
    """

    axis = -2

    theta *= -1
    co_axis_1 = -1
    co_axis_1_slice = (None, None, ...)
    co_axis_2 = -3
    co_axis_2_slice = (..., None, None)
    a = np.tan(theta / 2)
    b = -np.sin(theta)

    # phase_prefactor_a
    # phase_prefactor_b

    freq_c1 = fftshift_t(fftfreq_t(X.shape[co_axis_1], 1), dim=(0))[co_axis_1_slice]
    freq_c2 = fftshift_t(fftfreq_t(X.shape[co_axis_2], 1), dim=(0))[co_axis_2_slice]
    pix_1 = (th.linspace(-1, 1, X.shape[co_axis_1]) * (X.shape[co_axis_1] // 2) * b)[
        co_axis_1_slice
    ]
    pix_2 = (th.linspace(-1, 1, X.shape[co_axis_2]) * (X.shape[co_axis_2] // 2) * a)[
        co_axis_2_slice
    ]

    shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
    shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)

    return ifftnd_t(
        fftnd_t(
            ifftnd_t(
                fftnd_t(
                    ifftnd_t(fftnd_t(X, (co_axis_1)) * (shift_exp_c1), (co_axis_1)),
                    (co_axis_2),
                )
                * (shift_exp_c2),
                (co_axis_2),
            ),
            (co_axis_1),
        )
        * (shift_exp_c1),
        (co_axis_1),
    )


def rotate_3D_m3(X: th.Tensor, theta: float) -> th.Tensor:
    """
    Apply a 3D rotation to an input tensor around -3 axis.
    Args:
        X (Tensor): The input tensor to be rotated.
        theta (float): The rotation angle in radians.
    Returns:
        Tensor: The rotated tensor.
    """
    theta *= -1
    axis = -3

    co_axis_1 = -2
    co_axis_1_slice = (None, ..., None)
    co_axis_2 = -1
    co_axis_2_slice = (None, None, ...)
    a = np.tan(theta / 2)
    b = -np.sin(theta)

    # phase_prefactor_a
    # phase_prefactor_b

    freq_c1 = fftshift_t(fftfreq_t(X.shape[co_axis_1], 1), dim=(0))[co_axis_1_slice]
    freq_c2 = fftshift_t(fftfreq_t(X.shape[co_axis_2], 1), dim=(0))[co_axis_2_slice]
    pix_1 = (th.linspace(-1, 1, X.shape[co_axis_1]) * (X.shape[co_axis_1] // 2) * b)[
        co_axis_1_slice
    ]
    pix_2 = (th.linspace(-1, 1, X.shape[co_axis_2]) * (X.shape[co_axis_2] // 2) * a)[
        co_axis_2_slice
    ]

    shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
    shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)

    return ifftnd_t(
        fftnd_t(
            ifftnd_t(
                fftnd_t(
                    ifftnd_t(fftnd_t(X, (co_axis_1)) * (shift_exp_c1), (co_axis_1)),
                    (co_axis_2),
                )
                * (shift_exp_c2),
                (co_axis_2),
            ),
            (co_axis_1),
        )
        * (shift_exp_c1),
        (co_axis_1),
    )


### EXAMPLE ###

# test_cube = th.zeros(20, 20, 20, dtype=th.cfloat)
# test_cube[..., 5:15, 5:15, 5:15] = 1 - 1j

# X = test_cube
# X[9:,9:,9:]*=2


# plot_.show_3d(X)


# plot_.show_3d(rotate_3D_m1(X,-np.radians(10)))
# plot_.show_3d(rotate_3D_m1(X,np.radians(10)))


# plot_.show_3d(rotate_3D_m2(X,-np.radians(10)))
# plot_.show_3d(rotate_3D_m2(X,np.radians(10)))


# plot_.show_3d(rotate_3D_m3(X,-np.radians(10)))
# plot_.show_3d(rotate_3D_m3(X,np.radians(10)))

### END ###
