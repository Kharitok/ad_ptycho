"""
Shifter models for AD ptychography
"""
import torch.nn.functional as F
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader
# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Roration matricies___________


def get_rot_mat(theta):
    """Rotational matrix for the angle theta, conterclockwise from ^Y axis"""
    theta = th.tensor(theta)
    return th.tensor(
        [
            [th.cos(theta), -th.sin(theta), 0],
            [th.sin(theta), th.cos(theta), 0],
        ]
    )


def get_trans_mat(tx, ty):
    """Translational matrix for the translation of tx (+ =  <) ty (+ = ^)"""
    tx, ty = th.tensor(tx), th.tensor(ty)
    return th.tensor([[1, 0, tx], [0, 1, ty]])


def get_scaling_mat(scale_x, scale_y):
    """Scalling matrix for the scaling of 1/sx (+ =  >) 1/sy (+ = ^)"""
    sx, sy = th.tensor(scale_x), th.tensor(scale_y)
    return th.tensor([[sx, 0, 0], [0, sy, 0]])


def get_shear_mat(shear_x, shear_y):
    """Shear matrix for shear of  shear_x (tan phi measured clockwise from ^y) shear_y (tan phi measured
    counterclockwise from >x)"""
    shear_x, shear_y = th.tensor(shear_x), th.tensor(shear_y)
    return th.tensor([[1, shear_x, 0], [shear_y, 1, 0]])


# def Return_total_transform(Scale,Rot,Trans,Shear):
#     """constructs total affine transformation matrix from Translation, rotation, scaling, and Shear components"""
#     return()


# ___________Shifters___________


class ShifterDefault(th.nn.Module):
    """Scans the sample using differentiable affine transformation, performs position correction by optimizing xand y shifts
    init_shifts are assumed to be (N,2) array or tensor (N_pos,[x,y]) in pixels +y = |^, +x = <-
    """

    def __init__(self, init_shifts, borders, sample_size, mode="bilinear"):
        super().__init__()
        self.borders = borders
        self.sample_size = sample_size
        self.init_shifts = init_shifts * (
            2 / self.sample_size
        )  # convert shifts from pixels to affine grid units

        self.ze_l = th.zeros(len(init_shifts))
        self.one_l = th.ones(len(init_shifts))
        self.mode = mode

        self.register_buffer("shifts_initial", self.init_shifts.data)  # .cuda()
        self.register_buffer(
            "left_identity_matrix",
            th.stack(
                tensors=[
                    th.stack(tensors=[self.one_l, self.ze_l], dim=1),
                    th.stack(tensors=[self.ze_l, self.one_l], dim=1),
                ],
                dim=1,
            ),
        )

        self.shifts_correction = nn.Parameter(
            th.zeros((len(self.init_shifts), 2), dtype=th.float32)
        )
        # theta_result = th.stack(tensors=[th.stack(tensors=[cost,-1*sint,tx+tx_init],dim=1),
        #           th.stack(tensors=[sint,cost,ty+ty_init],dim=1)],dim=1)#good

    def forward(self, sample, scan_numbers):
        """get sample function at coordinates corresponding to scan_numbers"""
        full_theta = th.cat(
            [
                self.left_identity_matrix[scan_numbers],
                self.shifts_initial[scan_numbers, :, None]
                + self.shifts_correction[scan_numbers, :, None],
            ],
            dim=2,
        )

        # expand sample to the number of scan positions to perform shifting in paralell
        #         print('sample:',type(sample),sample.shape,'scan_numbers:',type(scan_numbers),scan_numbers.shape)
        sample = sample.expand(len(scan_numbers), -1, -1)
        # permute dimensions of sample viewed as 2 real valued tensors to get in in (N, C, H, W)  shape where C is 2 as  real imag

        sample = th.view_as_real(sample).permute(0, 3, 1, 2)

        grid = F.affine_grid(
            full_theta, sample.size(), align_corners=False
        )  # .cuda()???
        sample = F.grid_sample(
            sample, grid, padding_mode="zeros", mode=self.mode, align_corners=False
        )
        # permute shifted sample back  and then allign in the memory to have it contiguous for future operations
        return th.view_as_complex(sample.permute(0, 2, 3, 1).contiguous())[
            :, self.borders[0] : self.borders[1], self.borders[2] : self.borders[3]
        ]

    def get_coordinates_in_pixels(self):
        with th.no_grad():
            init_coords = self.init_shifts / 2 * self.sample_size
            corrected_coords = (
                (self.shifts_initial + self.shifts_correction) / 2 * self.sample_size
            )

        return {
            "initial_coordinates": init_coords,
            "corrected coordinates": corrected_coords,
        }


class ShifterElastic(ShifterDefault):
    """Scans the sample using differentiable affine transformation, performs position correction by optimizing xa nd y shifts
    scan positionsrefinement is  performed with the maximum correction limited by  max_correction (pixels) value.
    """

    def __init__(
        self, init_shifts, borders, sample_size, max_correction, mode="bilinear"
    ):
        super().__init__(init_shifts, borders, sample_size, mode=mode)

        self.register_buffer(
            "max_correction", th.tensor(max_correction * (2 / self.sample_size)).data
        )
        self.Tanh = th.nn.Tanh()

    def forward(self, sample, scan_numbers):
        # here  self.shifts_correction goes through function to be restricted within -1..1 and the nmultiplied by self.max_correction
        full_theta = th.cat(
            [
                self.left_identity_matrix[scan_numbers],
                self.shifts_initial[scan_numbers, :, None]
                + (self.Tanh(self.shifts_correction) * self.max_correction)[
                    scan_numbers, :, None
                ],
            ],
            dim=2,
        )

        # expand sample to the number of scan positions to perform shifting in paralell
        sample = sample.expand(len(scan_numbers), -1, -1)
        # permute dimensions of sample viewed as 2 real valued tensors to get in in (N, C, H, W)  shape where C is 2 as  real imag

        sample = th.view_as_real(sample).permute(0, 3, 1, 2)

        grid = F.affine_grid(
            full_theta, sample.size(), align_corners=False
        )  # .cuda()???
        sample = F.grid_sample(
            sample, grid, padding_mode="zeros", mode=self.mode, align_corners=False
        )
        # permute shifted sample back  and then allign in the memory to have it contiguous for future operations
        return (th.view_as_complex(sample.permute(0, 2, 3, 1).contiguous()))[
            :, self.borders[0] : self.borders[1], self.borders[2] : self.borders[3]
        ]

    def get_coordinates_in_pixels(self):
        with th.no_grad():
            init_coords = self.init_shifts / 2 * self.sample_size
            corrected_coords = (
                (
                    self.shifts_initial
                    + (self.Tanh(self.shifts_correction) * self.max_correction)
                )
                / 2
                * self.sample_size
            )

        return {
            "initial_coordinates": init_coords,
            "corrected coordinates": corrected_coords,
        }


class ShifterRotational(ShifterDefault):
    """Scans the sample using differentiable affine transformation, performs position correction by optimizing xand y shifts.
    In addition tries to correct for possible sample rotation at the each of the scan positions
    """

    def __init__(
        self,
        init_shifts,
        init_rotations,
        borders,
        sample_size,
    ):
        super().__init__(init_shifts, borders, sample_size, mode="bilinear")

        self.init_rotations = init_rotations
        self.register_buffer("rotations_initial", self.init_rotations.data)
        self.rotation_correction = nn.Parameter(
            th.zeros((len(self.init_rotations)), dtype=th.float32)
        )

    def forward(self, sample, scan_numbers):
        resulting_rotations = (
            self.rotation_correction[scan_numbers]
            + self.rotations_initial[scan_numbers]
        )
        sin_theta, cos_theta = th.sin(resulting_rotations), th.cos(resulting_rotations)
        rotational_mattr = th.stack(
            tensors=[
                th.stack(tensors=[cos_theta, -1 * sin_theta], dim=1),
                th.stack(tensors=[sin_theta, cos_theta], dim=1),
            ],
            dim=1,
        )  # good

        full_theta = th.cat(
            [
                rotational_mattr,
                self.shifts_initial[scan_numbers, :, None]
                + self.shifts_correction[scan_numbers, :, None],
            ],
            dim=2,
        )

        # expand sample to the number of scan positions to perform shifting in paralell
        sample = sample.expand(len(scan_numbers), -1, -1)
        # permute dimensions of sample viewed as 2 real valued tensors to get in in (N, C, H, W)  shape where C is 2 as  real imag

        sample = th.view_as_real(sample).permute(0, 3, 1, 2)

        grid = F.affine_grid(
            full_theta, sample.size(), align_corners=False
        )  # .cuda()???
        sample = F.grid_sample(
            sample, grid, padding_mode="zeros", mode=self.mode, align_corners=False
        )
        # permute shifted sample back  and then allign in the memory to have it contiguous for future operations
        return th.view_as_complex(sample.permute(0, 2, 3, 1).contiguous())[
            :, self.borders[0] : self.borders[1], self.borders[2] : self.borders[3]
        ]

    def get_coordinates_in_pixels(self):
        with th.no_grad():
            init_coords = self.init_shifts / 2 * self.sample_size
            corrected_coords = (
                (self.shifts_initial + self.shifts_correction) / 2 * self.sample_size
            )

            init_rot = self.init_rotations
            corrected_rot = self.rotation_correction + self.rotations_initial

        return {
            "initial_coordinates": init_coords,
            "corrected_coordinates": corrected_coords,
            "initial_rotations": init_rot,
            "corrected_rotations": corrected_rot,
        }


class ShifterRotationalElastic(ShifterRotational):
    """Scans the sample using differentiable affine transformation, performs position correction by optimizing xand y shifts.
    In addition tries to correct for possible sample rotation at the each of the scan positions
    scan positionsrefinement is  performed with the maximum correction limited by  max_correction (pixels) value.
    """

    def __init__(
        self,
        init_shifts,
        init_rotations,
        borders,
        sample_size,
        max_correction,
        max_rotation_correction,
    ):
        super().__init__(
            init_shifts,
            init_rotations,
            borders,
            sample_size,
        )

        self.init_rotations = init_rotations
        self.register_buffer("rotations_initial", self.init_rotations.data)
        self.rotations_correction = nn.Parameter(
            th.zeros((len(self.init_rotations)), dtype=th.float32)
        )
        self.register_buffer(
            "max_correction", th.tensor(max_correction * (2 / self.sample_size)).data
        )
        self.register_buffer("max_rotation", th.tensor(max_rotation_correction).data)
        self.Tanh = th.nn.Tanh()

    def forward(self, sample, scan_numbers):
        resulting_rotations = (
            self.Tanh(self.rotation_correction[scan_numbers]) * self.max_rotation
            + self.rotations_initial[scan_numbers]
        )
        sin_theta, cos_theta = th.sin(resulting_rotations), th.cos(resulting_rotations)
        rotational_mattr = th.stack(
            tensors=[
                th.stack(tensors=[cos_theta, -1 * sin_theta], dim=1),
                th.stack(tensors=[sin_theta, cos_theta], dim=1),
            ],
            dim=1,
        )  # good

        full_theta = th.cat(
            [
                rotational_mattr,
                self.shifts_initial[scan_numbers, :, None]
                + (self.Tanh(self.shifts_correction) * self.max_correction)[
                    scan_numbers, :, None
                ],
            ],
            dim=2,
        )

        # expand sample to the number of scan positions to perform shifting in paralell
        sample = sample.expand(len(scan_numbers), -1, -1)
        # permute dimensions of sample viewed as 2 real valued tensors to get in in (N, C, H, W)  shape where C is 2 as  real imag

        sample = th.view_as_real(sample).permute(0, 3, 1, 2)

        grid = F.affine_grid(
            full_theta, sample.size(), align_corners=False
        )  # .cuda()???
        sample = F.grid_sample(
            sample, grid, padding_mode="zeros", mode=self.mode, align_corners=False
        )
        # permute shifted sample back  and then allign in the memory to have it contiguous for future operations
        return th.view_as_complex(sample.permute(0, 2, 3, 1).contiguous())[
            :, self.borders[0] : self.borders[1], self.borders[2] : self.borders[3]
        ]

    def get_coordinates_in_pixels(self):
        with th.no_grad():
            init_coords = self.init_shifts / 2 * self.sample_size
            corrected_coords = (
                (self.shifts_initial + self.shifts_correction) / 2 * self.sample_size
            )

            init_rot = self.init_rotations
            corrected_rot = self.rotation_correction + self.rotations_initial

        return {
            "initial_coordinates": init_coords,
            "corrected_coordinates": corrected_coords,
            "initial_rotations": init_rot,
            "corrected_rotations": corrected_rot,
        }




class ShifterDefault_Fourrier(th.nn.Module):
    """Scans the sample using differentiable fourrier shoift, performs position correction by optimizing xand y shifts
    init_shifts are assumed to be (N,2) array or tensor (N_pos,[x,y]) in pixels +y = |^, +x = <-
    """

    def __init__(self, init_shifts, borders, sample_size,):
        super().__init__()
        self.borders = borders
#         self.sample_size = sample_size
        self.init_shifts = th.flip(init_shifts*-1,dims = [1]) #in pixels

        self.register_buffer("shifts_initial", self.init_shifts.data)  # .cuda()
        freq_x, freq_y = th.meshgrid(th.fft.fftfreq(sample_size, 1), th.fft.fftfreq(sample_size, 1),indexing = 'ij')
        self.register_buffer("freqs_x", freq_x.data)
        self.register_buffer("freqs_y", freq_y)
        
        self.shifts_correction = nn.Parameter(
            th.zeros((len(self.init_shifts), 2), dtype=th.float32)
        )

        # t_shift = th.tensor(((100,0),(0,0),(0,100)),dtype = th.float)
        # t_shift.requires_grad= True
        # f_signal

        # f_signal = th.fft.fft2(t_s,norm = 'ortho')
        # freq_x, freq_y = th.meshgrid(th.fft.fftfreq(f_signal.shape[0], 1), th.fft.fftfreq(f_signal.shape[1], 1),indexing = 'ij')
        # shift_op = th.exp(-2j*th.pi * (freq_x[None,...]*t_shift[:,0][...,None,None]+freq_y[None,...]*t_shift[:,1][...,None,None]))
        # shifted_f = shift_op*f_signal
        # shifted_signal = th.fft.ifft2(shifted_f,norm = 'ortho')

    def forward(self, sample, scan_numbers):
        """get sample function at coordinates corresponding to scan_numbers"""
        shifts = self.shifts_correction[scan_numbers]+self.shifts_initial[scan_numbers]

        
        sample = th.fft.fft2(sample,norm = 'ortho')
        sample = sample*th.exp(-2j*th.pi * (self.freqs_x[None,...]*shifts[:,0][...,None,None]+self.freqs_y[None,...]*shifts[:,1][...,None,None]))
        sample = th.fft.ifft2(sample,norm = 'ortho')
        
        return sample[
            :, self.borders[0] : self.borders[1], self.borders[2] : self.borders[3]
        ]

