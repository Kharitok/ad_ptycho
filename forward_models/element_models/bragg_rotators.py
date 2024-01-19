"""
Projectors - rotators for Bragg ptychography
"""

import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.fft import fftshift as fftshift_t,  ifftshift as ifftshift_t , fftn as fftn_t, ifftn as ifftn_t, fftfreq as fftfreq_t
import numpy as np



class Probe_3d_projector(th.nn.Module):
    """
    Extracts 2d probe into 3d by adding the third dimension 
    and then rotates it on double bragg angle
    to align propagation dimension with z axis
    """

    def __init__(
        self,
        interaction_length_pixels:int,
        probe_shape:th.Size,
        bragg_angle:float,
        axis:int = -2,


    ) -> None:
        
        super().__init__()
        
        self.interaction_length_pixels = interaction_length_pixels
        self.probe_shape = probe_shape
        self.probe_shape_3d = (interaction_length_pixels,)+probe_shape
        self.bragg_angle = bragg_angle
        self.axis = axis

        self.co_axis_1 = {-1:-3,-2:-1,-3:-2}.get(axis)
        self.co_axis_2 = {-1:-2,-2:-3,-3:-1}.get(axis)
        co_axis_1_slice = {-1:(...,None,None,),-2:(None, None, ...),-3:(None, ..., None)}.get(axis)
        co_axis_2_slice = {-1:(None,...,None),-2:(..., None, None),-3:(None, None, ...)}.get(axis)

        self.a = np.tan(bragg_angle / 2)
        self.b = -np.sin(bragg_angle)


        freq_c1 = fftshift_t(fftfreq_t(self.probe_shape_3d[self.co_axis_1], 1), dim=(0))[co_axis_1_slice]
        freq_c2 = fftshift_t(fftfreq_t(self.probe_shape_3d[self.co_axis_2], 1), dim=(0))[co_axis_2_slice]

        pix_1 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_1]) * (self.probe_shape_3d[self.co_axis_1] // 2) * self.b)[
        co_axis_1_slice]
        pix_2 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_2]) * (self.probe_shape_3d[self.co_axis_2] // 2) *self.a)[
        co_axis_2_slice]


        shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
        self.register_buffer("shift_exp_c1", shift_exp_c1.data)
        shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)
        self.register_buffer("shift_exp_c2", shift_exp_c2.data)


   


    def rotate(self,probe:th.Tensor) -> th.Tensor:
        return ifftnd_t(
            fftnd_t(
                ifftnd_t(
                    fftnd_t(
                        ifftnd_t(fftnd_t(probe, (self.co_axis_1)) * (self.shift_exp_c1), (self.co_axis_1)),
                        (self.co_axis_2),
                    )
                    * (self.shift_exp_c2),
                    (self.co_axis_2),
                ),
                (self.co_axis_1),
            )
            * (self.shift_exp_c1),
            (self.co_axis_1),
        )
    
    
    def forward(self,probe:th.Tensor) -> th.Tensor:
        return (self.rotate(self.rotate(probe.unsqueeze(-3))))
    


class Probe_3d_projector_reduced(th.nn.Module):
    """
    Extracts 2d probe into 3d by adding the third dimension 
    and then rotates it on double bragg angle
    to align propagation dimension with z axis
    """

    def __init__(
        self,
        interaction_length_pixels:int,
        probe_shape:th.Size,
        bragg_angle:float,
        axis:int = -2,


    ) -> None:
        
        super().__init__()
        
        self.interaction_length_pixels = interaction_length_pixels
        self.probe_shape = probe_shape
        self.probe_shape_3d = (interaction_length_pixels,)+probe_shape
        self.bragg_angle = bragg_angle
        self.axis = axis

        self.co_axis_1 = {-1:-3,-2:-1,-3:-2}.get(axis)
        self.co_axis_2 = {-1:-2,-2:-3,-3:-1}.get(axis)
        co_axis_1_slice = {-1:(...,None,None,),-2:(None, None, ...),-3:(None, ..., None)}.get(axis)
        co_axis_2_slice = {-1:(None,...,None),-2:(..., None, None),-3:(None, None, ...)}.get(axis)

        self.a = np.tan(bragg_angle / 2)
        self.b = -np.sin(bragg_angle)


        freq_c1 = (fftfreq_t(self.probe_shape_3d[self.co_axis_1], 1))[co_axis_1_slice]
        freq_c2 = (fftfreq_t(self.probe_shape_3d[self.co_axis_2], 1))[co_axis_2_slice]

        pix_1 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_1]) * (self.probe_shape_3d[self.co_axis_1] // 2) * self.b)[
        co_axis_1_slice]
        pix_2 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_2]) * (self.probe_shape_3d[self.co_axis_2] // 2) *self.a)[
        co_axis_2_slice]


        shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
        self.register_buffer("shift_exp_c1", shift_exp_c1.data)
        shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)
        self.register_buffer("shift_exp_c2", shift_exp_c2.data)


   


    def rotate(self,probe:th.Tensor) -> th.Tensor:
        return ifftn_t(
            fftn_t(
                ifftn_t(
                    fftn_t(
                        ifftn_t(fftn_t(probe, dim=(self.co_axis_1)) * (self.shift_exp_c1), dim=(self.co_axis_1)),
                        dim=(self.co_axis_2),
                    )
                    * (self.shift_exp_c2),
                    dim=(self.co_axis_2),
                ),
                dim=(self.co_axis_1),
            )
            * (self.shift_exp_c1),
            dim=(self.co_axis_1),
        )
    
    
    def forward(self,probe:th.Tensor) -> th.Tensor:
        return (self.rotate(self.rotate(probe.unsqueeze(-3))))