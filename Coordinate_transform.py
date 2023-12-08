from torch.fft import (
    fftshift as fftshift,
    fftn as fftn_t,
    ifftn as ifftn_t,
    ifftshift as ifftshift,
)
import torch.fft as th_fft
import torch as th
import numpy as np


# def fft1d( arr, n ):
#     return fftshift_t(
#         fftn_t(
#             fftshift_t(
#                 arr, dim=[n]
#             ),
#             dim=[n],
#             norm='ortho'
#         ),
#         dim=[n]
#     )

# def ifft1d( arr, n ):
#     return fftshift_t(
#         ifftn_t(
#             fftshift_t(
#                 arr, dim=[n]
#             ),
#             dim=[n],
#             norm='ortho'
#         ),
#         dim=[n]
#     )


def shear_term(axis1, axis2, a, b):
    return th.exp(2j * th.pi * (a * axis1 + b * axis_2))


# def shear


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


def get_pad_size(shape, sampling_init, sampling_desired):
    sampling_init = th.tensor(sampling_init[::-1])
    sampling_desired = th.tensor(sampling_desired[::-1])

    return (
        ((-1 / 2) * (shape * (1 - sampling_init / sampling_desired)))
        .int()
        .repeat_interleave(2)
    )


def Bulk_Resample(X, padding):
    return Pad(
        th_iff_sampling(Pad(th_ff_sampling(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )


def Bulk_Resample_3d(X, padding):
    return Pad(
        th_iff_sampling_3d(Pad(th_ff_sampling_3d(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )


# def shear_ramp(array,)


def shear(array, axis, magnitude):
    coord_1_ax = 1 if axis == 0 else 0
    coord_2_ax = 1 if axis == 2 else 2
    slice_cord_1 = [None] * coord_1_ax + [slice(None)] + [None] * (2 - coord_1_ax)
    slice_cord_2 = [None] * coord_2_ax + [slice(None)] + [None] * (2 - coord_2_ax)
    slice_freq = [None] * axis + [slice(None)] + [None] * (2 - axis)

    coord_1_ax_val = th.arange(array.shape[coord_1_ax]) - int(
        array.shape[coord_1_ax] // 2
    )
    coord_2_ax_val = th.arange(array.shape[coord_2_ax]) - int(
        array.shape[coord_2_ax] // 2
    )
    freqs = th.fft.fftfreq(array.shape[axis], 1)[slice_freq]

    coord_1_ax_val = coord_1_ax_val[slice_cord_1]
    coord_2_ax_val = coord_2_ax_val[slice_cord_2]

    sheared = th.fft.fft(array, dim=axis, norm="ortho")
    sheared = sheared * th.exp(
        -2j
        * th.pi
        * freqs
        * (magnitude[0] * coord_1_ax_val + magnitude[1] * coord_2_ax_val)
    )
    sheared = th.fft.ifft(sheared, dim=axis, norm="ortho")
    return sheared


def shear_transform(array, shear_axis, ramp_axis, magnitude):
    coord_1_ax, coord_2_ax = ramp_axis

    slice_cord_1 = [None] * coord_1_ax + [slice(None)] + [None] * (2 - coord_1_ax)
    slice_cord_2 = [None] * coord_2_ax + [slice(None)] + [None] * (2 - coord_2_ax)
    slice_freq = [None] * shear_axis + [slice(None)] + [None] * (2 - shear_axis)

    coord_1_ax_val = th.arange(array.shape[coord_1_ax]) - int(
        array.shape[coord_1_ax] // 2
    )
    coord_2_ax_val = th.arange(array.shape[coord_2_ax]) - int(
        array.shape[coord_2_ax] // 2
    )
    freqs = th.fft.fftfreq(sample.shape[shear_axis], 1)[slice_freq]

    coord_1_ax_val = coord_1_ax_val[slice_cord_1]
    coord_2_ax_val = coord_2_ax_val[slice_cord_2]

    sheared = th.fft.fft(array, dim=shear_axis, norm="ortho")
    sheared = sheared * th.exp(
        -2j
        * th.pi
        * freqs
        * (magnitude[0] * coord_1_ax_val + magnitude[1] * coord_2_ax_val)
    )
    sheared = th.fft.ifft(sheared, dim=shear_axis, norm="ortho")
    return sheared


# def rotate(array,axis,theta):

#     a

#     X = shear(X,(axis+1)%3
#     X = shear(X,(axis+1)%3
#     X = shear(X,(axis+1)%3

#     return X


def rotate(array, axis, angle):
    if axis == 0:
        X = shear(array, (axis + 1) % 3, (0, np.tan(angle / 2)))
        X = shear(X, (axis - 1) % 3, (0, -np.sin(angle)))
        return shear(X, (axis + 1) % 3, (0, np.tan(angle / 2)))
    elif axis == 1:
        X = shear(array, (axis + 1) % 3, (np.tan(angle / 2), 0))
        X = shear(X, (axis - 1) % 3, (0, -np.sin(angle)))
        return shear(X, (axis + 1) % 3, (np.tan(angle / 2), 0))

    elif axis == 2:
        X = shear(array, (axis + 1) % 3, (np.tan(angle / 2), 0))
        X = shear(X, (axis - 1) % 3, (-np.sin(angle), 0))
        return shear(X, (axis + 1) % 3, (np.tan(angle / 2), 0))


def shift(sample, shift):
    sample_shape = sample.shape
    if len(sample_shape) == 2:
        freq_x, freq_y = th.meshgrid(
            th.fft.fftfreq(sample_shape[0], 1),
            th.fft.fftfreq(sample_shape[1], 1),
            indexing="ij",
        )
        sample = th.fft.fftn(sample, dim=(-1, -2), norm="ortho")
        sample = sample * th.exp(-2j * th.pi * (freq_x * shift[0] + freq_y * shift[1]))
        sample = th.fft.fftn(sample, dim=(-1, -2), norm="ortho")

    else:
        freq_x, freq_y, freq_z = th.meshgrid(
            th.fft.fftfreq(sample_shape[0], 1),
            th.fft.fftfreq(sample_shape[1], 1),
            th.fft.fftfreq(sample_shape[2], 1),
            indexing="ij",
        )
        sample = th.fft.fftn(sample, dim=(-1, -2, -3), norm="ortho")
        sample = sample * th.exp(
            -2j * th.pi * (freq_x * shift[0] + freq_y * shift[1] + freq_z * shift[2])
        )
        sample = th.fft.fftn(sample, dim=(-1, -2, -3), norm="ortho")

    return sample


from torch.fft import (
    fftshift as fftshift,
    fftn as fftn_t,
    ifftn as ifftn_t,
    ifftshift as ifftshift,
)
import torch.fft as th_fft
import torch as th
import numpy as np


# def fft1d( arr, n ):
#     return fftshift_t(
#         fftn_t(
#             fftshift_t(
#                 arr, dim=[n]
#             ),
#             dim=[n],
#             norm='ortho'
#         ),
#         dim=[n]
#     )

# def ifft1d( arr, n ):
#     return fftshift_t(
#         ifftn_t(
#             fftshift_t(
#                 arr, dim=[n]
#             ),
#             dim=[n],
#             norm='ortho'
#         ),
#         dim=[n]
#     )


def shear_term(axis1, axis2, a, b):
    return th.exp(2j * th.pi * (a * axis1 + b * axis_2))


# def shear


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


def get_pad_size(shape, sampling_init, sampling_desired):
    sampling_init = th.tensor(sampling_init[::-1])
    sampling_desired = th.tensor(sampling_desired[::-1])

    return (
        ((-1 / 2) * (shape * (1 - sampling_init / sampling_desired)))
        .int()
        .repeat_interleave(2)
    )


def Bulk_Resample(X, padding):
    return Pad(
        th_iff_sampling(Pad(th_ff_sampling(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )


def Bulk_Resample_3d(X, padding):
    return Pad(
        th_iff_sampling_3d(Pad(th_ff_sampling_3d(X), tuple(padding.tolist()))),
        tuple((-1 * padding).tolist()),
    )


# def shear_ramp(array,)


def shear(array, axis, magnitude):
    coord_1_ax = 1 if axis == 0 else 0
    coord_2_ax = 1 if axis == 2 else 2
    slice_cord_1 = [None] * coord_1_ax + [slice(None)] + [None] * (2 - coord_1_ax)
    slice_cord_2 = [None] * coord_2_ax + [slice(None)] + [None] * (2 - coord_2_ax)
    slice_freq = [None] * axis + [slice(None)] + [None] * (2 - axis)

    coord_1_ax_val = th.arange(array.shape[coord_1_ax]) - int(
        array.shape[coord_1_ax] // 2
    )
    coord_2_ax_val = th.arange(array.shape[coord_2_ax]) - int(
        array.shape[coord_2_ax] // 2
    )
    freqs = th.fft.fftfreq(array.shape[axis], 1)[slice_freq]

    coord_1_ax_val = coord_1_ax_val[slice_cord_1]
    coord_2_ax_val = coord_2_ax_val[slice_cord_2]

    sheared = th.fft.fft(array, dim=axis, norm="ortho")
    sheared = sheared * th.exp(
        -2j
        * th.pi
        * freqs
        * (magnitude[0] * coord_1_ax_val + magnitude[1] * coord_2_ax_val)
    )
    sheared = th.fft.ifft(sheared, dim=axis, norm="ortho")
    return sheared


def shear_transform(array, shear_axis, ramp_axis, magnitude):
    coord_1_ax, coord_2_ax = ramp_axis

    slice_cord_1 = [None] * coord_1_ax + [slice(None)] + [None] * (2 - coord_1_ax)
    slice_cord_2 = [None] * coord_2_ax + [slice(None)] + [None] * (2 - coord_2_ax)
    slice_freq = [None] * shear_axis + [slice(None)] + [None] * (2 - shear_axis)

    coord_1_ax_val = th.arange(array.shape[coord_1_ax]) - int(
        array.shape[coord_1_ax] // 2
    )
    coord_2_ax_val = th.arange(array.shape[coord_2_ax]) - int(
        array.shape[coord_2_ax] // 2
    )
    freqs = th.fft.fftfreq(sample.shape[shear_axis], 1)[slice_freq]

    coord_1_ax_val = coord_1_ax_val[slice_cord_1]
    coord_2_ax_val = coord_2_ax_val[slice_cord_2]

    sheared = th.fft.fft(array, dim=shear_axis, norm="ortho")
    sheared = sheared * th.exp(
        -2j
        * th.pi
        * freqs
        * (magnitude[0] * coord_1_ax_val + magnitude[1] * coord_2_ax_val)
    )
    sheared = th.fft.ifft(sheared, dim=shear_axis, norm="ortho")
    return sheared


# def rotate(array,axis,theta):

#     a

#     X = shear(X,(axis+1)%3
#     X = shear(X,(axis+1)%3
#     X = shear(X,(axis+1)%3

#     return X


def rotate(array, axis, angle):
    if axis == 0:
        X = shear(array, (axis + 1) % 3, (0, np.tan(angle / 2)))
        X = shear(X, (axis - 1) % 3, (0, -np.sin(angle)))
        return shear(X, (axis + 1) % 3, (0, np.tan(angle / 2)))
    elif axis == 1:
        X = shear(array, (axis + 1) % 3, (np.tan(angle / 2), 0))
        X = shear(X, (axis - 1) % 3, (0, -np.sin(angle)))
        return shear(X, (axis + 1) % 3, (np.tan(angle / 2), 0))

    elif axis == 2:
        X = shear(array, (axis + 1) % 3, (np.tan(angle / 2), 0))
        X = shear(X, (axis - 1) % 3, (-np.sin(angle), 0))
        return shear(X, (axis + 1) % 3, (np.tan(angle / 2), 0))


def shift(sample, shift):
    sample_shape = sample.shape
    if len(sample_shape) == 2:
        freq_x, freq_y = th.meshgrid(
            th.fft.fftfreq(sample_shape[0], 1),
            th.fft.fftfreq(sample_shape[1], 1),
            indexing="ij",
        )
        sample = th.fft.fftn(sample, dim=(-1, -2), norm="ortho")
        sample = sample * th.exp(-2j * th.pi * (freq_x * shift[0] + freq_y * shift[1]))
        sample = th.fft.fftn(sample, dim=(-1, -2), norm="ortho")

    else:
        freq_x, freq_y, freq_z = th.meshgrid(
            th.fft.fftfreq(sample_shape[0], 1),
            th.fft.fftfreq(sample_shape[1], 1),
            th.fft.fftfreq(sample_shape[2], 1),
            indexing="ij",
        )
        sample = th.fft.fftn(sample, dim=(-1, -2, -3), norm="ortho")
        sample = sample * th.exp(
            -2j * th.pi * (freq_x * shift[0] + freq_y * shift[1] + freq_z * shift[2])
        )
        sample = th.fft.fftn(sample, dim=(-1, -2, -3), norm="ortho")

    return sample
