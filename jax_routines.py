"""
This module contains jax functions for shifts and rotation of  2d and 3d 
Jax tensors based on fft transform
"""

import jax
import jax.numpy as jnp

# import numpy as np


def fft2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the two-dimensional Fast Fourier Transform (FFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The FFT of the input array.
    """
    return jnp.fft.fft2(x, norm="ortho", axes=(-1, -2))


def ifft2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the two-dimensional inverse Fast Fourier Transform (IFFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The IFFT of the input array.
    """
    return jnp.fft.ifft2(x, norm="ortho", axes=(-1, -2))


def FT2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the centered two-dimensional Fast Fourier Transform (FFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The centered FFT of the input array.
    """
    return jnp.fft.fftshift(
        jnp.fft.fft2(jnp.fft.ifftshift(x, axes=(-1, -2)), norm="ortho", axes=(-1, -2)),
        axes=(-1, -2),
    )


def IFT2(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the centered two-dimensional inverse Fast Fourier Transform (IFFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The centered IFFT of the input array.
    """
    return jnp.fft.fftshift(
        jnp.fft.ifft2(jnp.fft.ifftshift(x, axes=(-1, -2)), norm="ortho", axes=(-1, -2)),
        axes=(-1, -2),
    )


def fft3(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the three-dimensional Fast Fourier Transform (FFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The FFT of the input array.
    """
    return jnp.fft.fftn(x, norm="ortho", axes=(-1, -2, -3))


def ifft3(x: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the centered three-dimensional inverse Fast Fourier Transform (IFFT) of a given input array.

    Parameters:
    x (jnp.ndarray): The input array to be transformed.

    Returns:
    jnp.ndarray: The IFFT of the input array.
    """
    return jnp.fft.ifftn(x, norm="ortho", axes=(-1, -2, -3))


def get_freqs(x: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Computes the frequencies of the given input array along a specified dimension.

    Parameters:
        x (jnp.ndarray): The input array.
        dim (int): The dimension along which to compute the frequencies.

    Returns:
        jnp.ndarray: The frequencies of the input array along the specified dimension.
    """
    return jnp.fft.fftfreq(x.shape[dim])


def get_pix_num_for_shear(x: jnp.ndarray, dim: int) -> jnp.ndarray:
    """
    Computes the pixel numbers for shearing along a specified dimension.

    Parameters:
        x (jnp.ndarray): The input array.
        dim (int): The dimension along which to compute the pixel numbers.

    Returns:
        jnp.ndarray: The pixel numbers for shearing along the specified dimension.
    """
    return jnp.linspace(-x.shape[dim] // 2, x.shape[dim] // 2, x.shape[dim])


####
def shift_2d_tensor(
    x: jnp.ndarray,
    d0: jnp.ndarray,
    d1: jnp.ndarray,
    freqs_0: jnp.ndarray,
    freqs_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shifts a 2D tensor by applying a phase shift in the frequency domain.

    Parameters:
        x (jnp.ndarray): The input 2D tensor to be shifted.
        d0 (jnp.ndarray): The shift amount along the first dimension.
        d1 (jnp.ndarray): The shift amount along the second dimension.
        freqs_0 (jnp.ndarray): The frequencies along the first dimension.
        freqs_1 (jnp.ndarray): The frequencies along the second dimension.

    Returns:
        jnp.ndarray: The shifted 2D tensor.
    """

    phase_shift = jnp.exp(
        -2j * jnp.pi * (freqs_0[..., None] * d0 + freqs_1[None, ...] * d1)
    )
    return ifft2(fft2(x) * phase_shift)


shift_2d_tensor_vect_shifts = jax.vmap(
    shift_2d_tensor, in_axes=(None, 0, 0, None, None)
)
shift_2d_tensor_vect_tens = jax.vmap(
    shift_2d_tensor, in_axes=(0, None, None, None, None)
)


def shift_2d_tensor_stack(
    x: jnp.ndarray,
    d0: jnp.ndarray,
    d1: jnp.ndarray,
    freqs_0: jnp.ndarray,
    freqs_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shifts a 3D stack of 2D tensors by applying a phase shift in the frequency domain.

    Parameters:
        x (jnp.ndarray): The input 3D stack of 2D tensors to be shifted.
        d0 (jnp.ndarray): The shift amount along the first dimension.
        d1 (jnp.ndarray): The shift amount along the second dimension.
        freqs_0 (jnp.ndarray): The frequencies along the first dimension.
        freqs_1 (jnp.ndarray): The frequencies along the second dimension.

    Returns:
        jnp.ndarray: The shifted 3D stack of 2D tensors.
    """

    phase_shift = jnp.exp(
        -2j
        * jnp.pi
        * (
            freqs_0[None, ..., None] * d0[:, None, None]
            + freqs_1[None, None, ...] * d1[:, None, None]
        )
    )
    return ifft2(fft2(x) * phase_shift)


def shift_3d_tensor_stack(
    x: jnp.ndarray,
    d0: jnp.ndarray,
    d1: jnp.ndarray,
    d2: jnp.ndarray,
    freqs_0: jnp.ndarray,
    freqs_1: jnp.ndarray,
    freqs_2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shifts a 4D stack of 3D tensors by applying a phase shift in the frequency domain.

    Parameters:
        x (jnp.ndarray): The input 4D stack of 3D tensors to be shifted.
        d0 (jnp.ndarray): The shift amount along the first dimension.
        d1 (jnp.ndarray): The shift amount along the second dimension.
        d2 (jnp.ndarray): The shift amount along the third dimension.
        freqs_0 (jnp.ndarray): The frequencies along the first dimension.
        freqs_1 (jnp.ndarray): The frequencies along the second dimension.
        freqs_2 (jnp.ndarray): The frequencies along the third dimension.

    Returns:
        jnp.ndarray: The shifted 4D stack of 3D tensors.
    """

    phase_shift = jnp.exp(
        -2j
        * jnp.pi
        * (
            freqs_0[None, ..., None, None] * d0[:, None, None, None]
            + freqs_1[None, None, ..., None] * d1[:, None, None, None]
            + freqs_2[None, None, None, ...] * d2[:, None, None, None]
        )
    )
    # print(phase_shift.shape,(freqs_0[None,...,None,None]*d0[:,None,None,None]).shape,
    # (freqs_1[None,None,...,None]*d1[:,None,None,None]).shape,
    # (freqs_2[None,None,None,...]*d2[:,None,None,None]).shape)
    return ifft3(fft3(x) * phase_shift)


jit_shift_3d_tensor_stack = jax.jit(shift_3d_tensor_stack)


def shift_3d_tensor_stack_seq(
    x: jnp.ndarray,
    d0: jnp.ndarray,
    d1: jnp.ndarray,
    d2: jnp.ndarray,
    freqs_0: jnp.ndarray,
    freqs_1: jnp.ndarray,
    freqs_2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shifts a 4D stack of 3D tensors by applying a phase shift in the frequency domain.

    Parameters:
        x (jnp.ndarray): The input 4D stack of 3D tensors to be shifted.
        d0 (jnp.ndarray): The shift amount along the first dimension.
        d1 (jnp.ndarray): The shift amount along the second dimension.
        d2 (jnp.ndarray): The shift amount along the third dimension.
        freqs_0 (jnp.ndarray): The frequencies along the first dimension.
        freqs_1 (jnp.ndarray): The frequencies along the second dimension.
        freqs_2 (jnp.ndarray): The frequencies along the third dimension.

    Returns:
        jnp.ndarray: The shifted 4D stack of 3D tensors.
    """

    # phase_shift = jnp.exp(
    #     -2j
    #     * jnp.pi
    #     * (
    #         freqs_0[None, ..., None, None] * d0[:, None, None, None]
    #         + freqs_1[None, None, ..., None] * d1[:, None, None, None]
    #         + freqs_2[None, None, None, ...] * d2[:, None, None, None]
    #     )
    # )
    # print(phase_shift.shape,(freqs_0[None,...,None,None]*d0[:,None,None,None]).shape,(freqs_1[None,None,...,None]*d1[:,None,None,None]).shape,(freqs_2[None,None,None,...]*d2[:,None,None,None]).shape)
    return ifft3(
        fft3(x)
        * jnp.exp(
            -2j * jnp.pi * freqs_0[None, ..., None, None] * d0[:, None, None, None]
        )
        * jnp.exp(
            -2j * jnp.pi * freqs_1[None, None, ..., None] * d1[:, None, None, None]
        )
        * jnp.exp(
            -2j * jnp.pi * freqs_2[None, None, None, ...] * d2[:, None, None, None]
        )
    )


def shift_3d_tensor_stack_seq_1d(
    x: jnp.ndarray,
    d0: jnp.ndarray,
    d1: jnp.ndarray,
    d2: jnp.ndarray,
    freqs_0: jnp.ndarray,
    freqs_1: jnp.ndarray,
    freqs_2: jnp.ndarray,
) -> jnp.ndarray:
    """
    Shifts a 3D tensor stack in the frequency domain using sequential 1D FFTs.

    Parameters:
        x (jnp.ndarray): The input 4D tensor stack of 3D tensors.
        d0 (jnp.ndarray): The shift values along the first axis.
        d1 (jnp.ndarray): The shift values along the second axis.
        d2 (jnp.ndarray): The shift values along the third axis.
        freqs_0 (jnp.ndarray): The frequency values along the first axis.
        freqs_1 (jnp.ndarray): The frequency values along the second axis.
        freqs_2 (jnp.ndarray): The frequency values along the third axis.

    Returns:
        jnp.ndarray: The shifted 4D stack of 3D tensors.
    """

    # phase_shift = jnp.exp(
    #     -2j
    #     * jnp.pi
    #     * (
    #         freqs_0[None, ..., None, None] * d0[:, None, None, None]
    #         + freqs_1[None, None, ..., None] * d1[:, None, None, None]
    #         + freqs_2[None, None, None, ...] * d2[:, None, None, None]
    #     )
    # )
    # print(phase_shift.shape,(freqs_0[None,...,None,None]*d0[:,None,None,None]).shape,(freqs_1[None,None,...,None]*d1[:,None,None,None]).shape,(freqs_2[None,None,None,...]*d2[:,None,None,None]).shape)

    x = jnp.fft.ifft(
        jnp.fft.fft(x, axis=1, norm="ortho")
        * jnp.exp(
            -2j * jnp.pi * freqs_0[None, ..., None, None] * d0[:, None, None, None]
        ),
        axis=1,
        norm="ortho",
    )
    x = jnp.fft.ifft(
        jnp.fft.fft(x, axis=2, norm="ortho")
        * jnp.exp(
            -2j * jnp.pi * freqs_1[None, None, ..., None] * d1[:, None, None, None]
        ),
        axis=2,
        norm="ortho",
    )
    x = jnp.fft.ifft(
        jnp.fft.fft(x, axis=3, norm="ortho")
        * jnp.exp(
            -2j * jnp.pi * freqs_2[None, None, None, ...] * d2[:, None, None, None]
        ),
        axis=3,
        norm="ortho",
    )
    return x


jit_shift_3d_tensor_stack = jax.jit(shift_3d_tensor_stack)
jit_shift_3d_tensor_stack_seq = jax.jit(shift_3d_tensor_stack_seq)
jit_shift_3d_tensor_stack_seq_1d = jax.jit(shift_3d_tensor_stack_seq_1d)


def shear_0(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 2D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    # axis = (-2,)
    # co_axis = -1
    axis_slice = (..., None)
    co_axis_slice = (None, ...)

    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[axis_slice] * (pix_x * magnitude)[co_axis_slice]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=-2, norm="ortho") * (shift_exp), axis=-2, norm="ortho"
    )


def shear_1(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 2D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    # axis = (-1,)
    # co_axis = -2
    axis_slice = (None, ...)
    co_axis_slice = (..., None)

    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[axis_slice] * (pix_x * magnitude)[co_axis_slice]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=-1, norm="ortho") * (shift_exp), axis=-1, norm="ortho"
    )


def shear_01(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 0
    # co_axis = 1

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[:, None, None] * (pix_x * magnitude)[None, :, None]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


def shear_10(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 1
    # co_axis = 0

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[None, :, None] * (pix_x * magnitude)[:, None, None]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


def shear_02(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 0
    # co_axis = 2

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[:, None, None] * (pix_x * magnitude)[None, None, :]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


def shear_20(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 2
    # co_axis = 0

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[None, None, :] * (pix_x * magnitude)[:, None, None]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


def shear_12(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 1
    # co_axis = 2

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[None, :, None] * (pix_x * magnitude)[None, None, :]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


def shear_21(
    x: jnp.ndarray, magnitude: float, freq_x: jnp.ndarray, pix_x: jnp.ndarray
) -> jnp.ndarray:
    """
    Shear 3D tensor along  0 axis
    X tensor to shear
    magnitude float in [-1..1] determines  the strength of shear
    """

    axis = 2
    # co_axis = 1

    # freq_x[:, None,None]
    # pix_x = (
    #     th.linspace(-X.shape[co_axis] // 2, X.shape[co_axis] // 2, X.shape[co_axis]) * magnitude
    # )[None, :,None]
    shift_exp = jnp.exp(
        -2j * jnp.pi * freq_x[None, None, :] * (pix_x * magnitude)[None, :, None]
    )
    return jnp.fft.ifft(
        jnp.fft.fft(x, axis=axis, norm="ortho") * shift_exp, axis=axis, norm="ortho"
    )


###
def rotate_0(
    x: jnp.ndarray,
    theta: float,
    freq_0: jnp.ndarray,
    pix_0: jnp.ndarray,
    freq_1: jnp.ndarray,
    pix_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rotate 3D tensor along  0 axis
    X tensor to shear
    theta angle in radians
    """
    a = jnp.tan(theta / 2)
    b = -jnp.sin(theta)

    x_rotated = shear_21(x, a, freq_0, pix_0)
    x_rotated = shear_12(x_rotated, b, freq_1, pix_1)
    x_rotated = shear_21(x_rotated, a, freq_0, pix_0)
    return x_rotated


def rotate_1(
    x: jnp.ndarray,
    theta: float,
    freq_0: jnp.ndarray,
    pix_0: jnp.ndarray,
    freq_1: jnp.ndarray,
    pix_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rotate 3D tensor along  0 axis
    X tensor to shear
    theta angle in radians
    """
    a = jnp.tan(theta / 2)
    b = -jnp.sin(theta)

    x_rotated = shear_20(x, a, freq_0, pix_0)
    x_rotated = shear_02(x_rotated, b, freq_1, pix_1)
    x_rotated = shear_20(x_rotated, a, freq_0, pix_0)
    return x_rotated


def rotate_2(
    x: jnp.ndarray,
    theta: float,
    freq_0: jnp.ndarray,
    pix_0: jnp.ndarray,
    freq_1: jnp.ndarray,
    pix_1: jnp.ndarray,
) -> jnp.ndarray:
    """
    Rotate 3D tensor along  0 axis
    X tensor to shear
    theta angle in radians
    """
    a = jnp.tan(theta / 2)
    b = -jnp.sin(theta)

    x_rotated = shear_10(x, a, freq_0, pix_0)
    x_rotated = shear_01(x_rotated, b, freq_1, pix_1)
    x_rotated = shear_10(x_rotated, a, freq_0, pix_0)
    return x_rotated
