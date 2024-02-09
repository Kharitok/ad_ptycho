"""
Projectors - rotators for Bragg ptychography
"""

import torch.nn.functional as F
import torch.nn as nn
import torch as th
from torch.fft import fftshift as fftshift_t,  ifftshift as ifftshift_t , fftn as fftn_t, ifftn as ifftn_t, fftfreq as fftfreq_t
import numpy as np
from   ad_ptycho.Fourrier_resampling_3d import  fftnd_t, ifftnd_t,rotate_3D_m1,rotate_3D_m2,rotate_3D_m3,shift_3d_fourrier



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



class Probe_3d_projector_reduced_near90(th.nn.Module):
    """
    Extracts 2d probe into 3d by adding the third dimension 
    and then rotates it on double bragg angle
    first by rotation on 90 and than by back rotation on (90-2 * bragg angle)
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
        self.co_axis_1 = {-1:-3,-2:-1,-3:-2}.get(axis)
        self.co_axis_2 = {-1:-2,-2:-3,-3:-1}.get(axis)
        co_axis_1_slice = {-1:(...,None,None,),-2:(None, None, ...),-3:(None, ..., None)}.get(axis)
        co_axis_2_slice = {-1:(None,...,None),-2:(..., None, None),-3:(None, None, ...)}.get(axis)
        
        self.interaction_length_pixels = interaction_length_pixels
        self.probe_shape = probe_shape
        
        t_shape = [interaction_length_pixels,]+list(probe_shape)
        # print(t_shape)
        tmp = t_shape[self.co_axis_1]
        t_shape[self.co_axis_1] = t_shape[self.co_axis_2]
        t_shape[self.co_axis_2] = tmp
    
        self.probe_shape_3d = tuple(t_shape)
        # print(self.probe_shape_3d)
        
        
        self.rotation_angle = -(np.radians(90)-2*bragg_angle)
        self.bragg_angle = bragg_angle
        self.axis = axis
        
        self.shape_for_pad = ((probe_shape[0] - interaction_length_pixels)//2,)*2
        


        


        self.a = np.tan(self.rotation_angle / 2)
        self.b = -np.sin(self.rotation_angle)


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

        self.reverse_ind  = th.arange(self.probe_shape_3d[self.co_axis_2]- 1, -1, -1).to(th.long)
        self.register_buffer("reverse_index", self.reverse_ind.data)

        self.register_buffer("dimproj", th.ones(self.interaction_length_pixels)[:,None,None].data)
   


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

        return self.rotate(th.transpose( probe.unsqueeze(-3)*self.dimproj,self.co_axis_1,self.co_axis_2)[:,self.reverse_ind,...])
                # Pad(self.shape_for_pad)
        # return (self.rotate(self.rotate(probe.unsqueeze(-3))))
    

class Probe_3d_projector_reduced_near90_cut(th.nn.Module):
    """
    Extracts 2d probe into 3d by adding the third dimension 
    and then rotates it on double bragg angle
    first by rotation on 90 and than by back rotation on (90-2 * bragg angle)
    to align propagation dimension with z axis
    """

    def __init__(
        self,
        sup_along,
        sup_hor,
        sup_ver,
        interaction_length_pixels:int,
        probe_shape:th.Size,
        bragg_angle:float,
        axis:int = -2,
        cut = 10,



    ) -> None:
        
        super().__init__()
        self.co_axis_1 = {-1:-3,-2:-1,-3:-2}.get(axis)
        self.co_axis_2 = {-1:-2,-2:-3,-3:-1}.get(axis)
        co_axis_1_slice = {-1:(...,None,None,),-2:(None, None, ...),-3:(None, ..., None)}.get(axis)
        co_axis_2_slice = {-1:(None,...,None),-2:(..., None, None),-3:(None, None, ...)}.get(axis)
        
        self.interaction_length_pixels = interaction_length_pixels
        self.probe_shape = probe_shape
        
        t_shape = [interaction_length_pixels,]+list(probe_shape)
        # print(t_shape)
        tmp = t_shape[self.co_axis_1]
        t_shape[self.co_axis_1] = t_shape[self.co_axis_2]
        t_shape[self.co_axis_2] = tmp
    
        self.probe_shape_3d = tuple(t_shape)
        # print(self.probe_shape_3d)
        
        
        self.rotation_angle = -(np.radians(90)-2*bragg_angle)
        self.bragg_angle = bragg_angle
        self.axis = axis
        
        self.shape_for_pad = ((probe_shape[0] - interaction_length_pixels)//2,)*2
        


        


        self.a = np.tan(self.rotation_angle / 2)
        self.b = -np.sin(self.rotation_angle)


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

        self.reverse_ind  = th.arange(self.probe_shape_3d[self.co_axis_2]- 1, -1, -1).to(th.long)
        self.register_buffer("reverse_index", self.reverse_ind.data)

        self.register_buffer("dimproj", th.ones(self.interaction_length_pixels)[:,None,None].data)
        # self.dimproj[:cut]=0
        # self.dimproj[-cut:]=0

        self.register_buffer("axn1", sup_ver[None,None,:,None].data)
        self.register_buffer("axn2", sup_hor[None,:,None,None].data)
        self.register_buffer("axn3", sup_along[None,None,None,:].data)


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
        # print(self.axn1.shape,self.axn2.shape,self.axn3.shape,self.rotate(th.transpose( probe.unsqueeze(-3)*self.dimproj,self.co_axis_1,self.co_axis_2)[:,self.reverse_ind,...]).shape)
        return (self.rotate(th.transpose( probe.unsqueeze(-3)*self.dimproj,self.co_axis_1,self.co_axis_2)[:,self.reverse_ind,...])*self.axn1
                *self.axn1*self.axn2*self.axn3)
                # *self.axn2*self.axn3[])
                # *self.axn3[])
                # Pad(self.shape_for_pad)
        # return (self.rotate(self.rotate(probe.unsqueeze(-3))))

# class Probe_3d_projector_reduced_near90_cut(th.nn.Module):
#     """
#     Extracts 2d probe into 3d by adding the third dimension 
#     and then rotates it on double bragg angle
#     first by rotation on 90 and than by back rotation on (90-2 * bragg angle)
#     to align propagation dimension with z axis
#     """

#     def __init__(
#         self,
#         interaction_length_pixels:int,
#         probe_shape:th.Size,
#         bragg_angle:float,
#         axis:int = -2,
#         cut = 10,


#     ) -> None:
        
#         super().__init__()
#         self.co_axis_1 = {-1:-3,-2:-1,-3:-2}.get(axis)
#         self.co_axis_2 = {-1:-2,-2:-3,-3:-1}.get(axis)
#         co_axis_1_slice = {-1:(...,None,None,),-2:(None, None, ...),-3:(None, ..., None)}.get(axis)
#         co_axis_2_slice = {-1:(None,...,None),-2:(..., None, None),-3:(None, None, ...)}.get(axis)
        
#         self.interaction_length_pixels = interaction_length_pixels
#         self.probe_shape = probe_shape
        
#         t_shape = [interaction_length_pixels,]+list(probe_shape)
#         # print(t_shape)
#         tmp = t_shape[self.co_axis_1]
#         t_shape[self.co_axis_1] = t_shape[self.co_axis_2]
#         t_shape[self.co_axis_2] = tmp
    
#         self.probe_shape_3d = tuple(t_shape)
#         # print(self.probe_shape_3d)
        
        
#         self.rotation_angle = -(np.radians(90)-2*bragg_angle)
#         self.bragg_angle = bragg_angle
#         self.axis = axis
        
#         self.shape_for_pad = ((probe_shape[0] - interaction_length_pixels)//2,)*2
        


        


#         self.a = np.tan(self.rotation_angle / 2)
#         self.b = -np.sin(self.rotation_angle)


#         freq_c1 = (fftfreq_t(self.probe_shape_3d[self.co_axis_1], 1))[co_axis_1_slice]
#         freq_c2 = (fftfreq_t(self.probe_shape_3d[self.co_axis_2], 1))[co_axis_2_slice]

#         pix_1 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_1]) * (self.probe_shape_3d[self.co_axis_1] // 2) * self.b)[
#         co_axis_1_slice]
#         pix_2 = (th.linspace(-1, 1, self.probe_shape_3d[self.co_axis_2]) * (self.probe_shape_3d[self.co_axis_2] // 2) *self.a)[
#         co_axis_2_slice]


#         shift_exp_c1 = th.exp(-2j * th.pi * freq_c1 * pix_2)
#         self.register_buffer("shift_exp_c1", shift_exp_c1.data)
#         shift_exp_c2 = th.exp(-2j * th.pi * freq_c2 * pix_1)
#         self.register_buffer("shift_exp_c2", shift_exp_c2.data)

#         self.reverse_ind  = th.arange(self.probe_shape_3d[self.co_axis_2]- 1, -1, -1).to(th.long)
#         self.register_buffer("reverse_index", self.reverse_ind.data)

#         self.register_buffer("dimproj", th.ones(self.interaction_length_pixels)[:,None,None].data)
#         # self.dimproj[:cut]=0
#         # self.dimproj[-cut:]=0

#         self.register_buffer("ax0", th.ones(self.interaction_length_pixels)[:,None,None].data)
#         self.register_buffer("ax1", th.ones(self.interaction_length_pixels)[:,None,None].data)
#         self.register_buffer("ax2", th.ones(self.interaction_length_pixels)[:,None,None].data)


#     def rotate(self,probe:th.Tensor) -> th.Tensor:
#         return ifftn_t(
#             fftn_t(
#                 ifftn_t(
#                     fftn_t(
#                         ifftn_t(fftn_t(probe, dim=(self.co_axis_1)) * (self.shift_exp_c1), dim=(self.co_axis_1)),
#                         dim=(self.co_axis_2),
#                     )
#                     * (self.shift_exp_c2),
#                     dim=(self.co_axis_2),
#                 ),
#                 dim=(self.co_axis_1),
#             )
#             * (self.shift_exp_c1),
#             dim=(self.co_axis_1),
#         )
    
    
#     def forward(self,probe:th.Tensor) -> th.Tensor:

#         return self.rotate(th.transpose( probe.unsqueeze(-3)*self.dimproj,self.co_axis_1,self.co_axis_2)[:,self.reverse_ind,...])
#                 # Pad(self.shape_for_pad)
#         # return (self.rotate(self.rotate(probe.unsqueeze(-3))))
    



def generate_slab_support(space_3d_shape,eta_rad,assumed_sample_thickness,rec_resolution):


    size = space_3d_shape


    z,y,x = np.arange(size[0])-size[0]//2 , np.arange(size[1])-size[1]//2, np.arange(size[2])-size[2]//2
    z,y,x = z*rec_resolution,y*rec_resolution,x*rec_resolution

    zz,xx,yy = np.meshgrid(z,y,x,sparse=True, indexing='ij')


    center_coords = np.array((0,0,0))
    normal_vector = np.array((np.sin(eta_rad),np.cos(eta_rad),0))
    shift_vec = np.array(((assumed_sample_thickness/2)/np.sin(eta_rad),0,0))

    center_coords_t = center_coords +shift_vec
    center_coords_b = center_coords -shift_vec

    # center_coords_t*=10
    if_plane_top = normal_vector[0]*zz+normal_vector[1]*yy+normal_vector[2]*xx  - np.sum(normal_vector*center_coords_t)
    if_plane_bottom = normal_vector[0]*zz+normal_vector[1]*yy+normal_vector[2]*xx  - np.sum(normal_vector*center_coords_b)

    return (if_plane_bottom>0)*(if_plane_top<0)