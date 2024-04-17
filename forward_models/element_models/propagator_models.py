"""
Contains propagators required for the AD-based ptychography
"""
import torch.nn.functional as F
# import torch.nn as nn
import torch as th
import torch.fft as th_fft
import numpy as np
from torch.fft import (
    fftshift as fftshift_t,
    ifftshift as ifftshift_t,
    fftn as fftn_t,
    ifftn as ifftn_t,
    fftfreq as fftfreq_t,
)
# from torch.utils.data import Dataset, DataLoader

th.pi = np.pi  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True

# ___________Fourrier transform staff___________


def th_ff(field):
    """FFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.fft2(th_fft.ifftshift(field, dim=(-1, -2)), norm="backward"), dim=(-1, -2)
    )


def th_iff(field):
    """IFFT routine for centered data"""
    return th_fft.fftshift(
        th_fft.ifft2(th_fft.ifftshift(field, dim=(-1, -2)), norm="backward"), dim=(-1, -2)
    )

def fftnd_t(X: th.Tensor, n: int) -> th.Tensor:
    """1d FFT on a tensor using torch.fft"""
    return fftshift_t(fftn_t(ifftshift_t(X, dim=n), dim=n, norm="ortho"), dim=n)


def ifftnd_t(X, n):
    """1d FFT on a tensor using torch.fft"""
    return fftshift_t(ifftn_t(ifftshift_t(X, dim=n), dim=n, norm="ortho"), dim=n)

# ___________Misc functions for propagators construction___________


def grid(
    pixel_size,
    pixel_num,
):
    """
    return grid of the field as a meshgrid
    """
    dx, dy = pixel_size, pixel_size
    (Nx,Ny) = pixel_num, pixel_num

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


class PropagatorFresnelSingleTransformFLuxPreserving(th.nn.Module):
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
        self.z = z
        self.pixel_size = pixel_size


    def forward(self, X):
        """Performs forward propagation"""
        return self.mul1 * th_ff(X * self.mul2) / self.num

    def inverse(self, X):
        """Performs inverse propagation"""
        return self.mul1_inv * th_iff(X * self.mul2_inv) * self.num


class PropagatorFraunhFluxPreserving(th.nn.Module):
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
        """Performs forward propagation"""
        return th_ff(X) / self.num

    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff(X) * self.num






def pad2_(x:th.Tensor,width):
    # width = x.shape[-1]//2
    return F.pad(x,(width,width,width,width),mode='constant',value=0)

def upsamp_f_(x:th.Tensor,width):
    return ifftnd_t(pad2_(fftnd_t(x,(-1,-2)),width),(-1,-2))


class PropagatorFraunhUpsampFluxPreserving(th.nn.Module):
    """
    Performs upsampled fraunhoffer propagation to the detector plane
    """

    def __init__(
        self,
        pixel_num,
    ):
        super().__init__()
        self.width = pixel_num//2
        self.w3 = self.width*3

    def forward(self, beam,diff):
        """Performs forward propagation"""
        return fftnd_t(upsamp_f_(beam,self.width)*upsamp_f_(diff,self.width),(-1,-2))[...,self.width:self.w3,self.width:self.w3]*2






# class PropagatorRSIR(th.nn.Module):
#     """
   
#     Rayleigh–Sommerfeld impulse responce propagation 
#     better for longer distances
#     Δx <= λz/L
#     """

#     def __init__(
#         self,
#         pixel_num,
#         pix_size,
#         wavelength,
#         z,
#     ):
#         super().__init__()

#         self.num = pixel_num
#         self.pix_size = pix_size
#         self.wavelength = wavelength
#         self.z = z
#         self.k = 2*th.pi/wavelength

  

#         xx, yy = grid(pixel_size=self.pix_size,pixel_num=self.num)


#         h = (self.z/(1j*self.wavelength)) * ((th.exp(1j*self.k*th.sqrt(self.z**2+xx**2+yy**2)))/(self.z**2+xx**2+yy**2))
#         H = th_ff(h)*self.pix_size**2

#         h_i = (-self.z/(1j*self.wavelength)) * ((th.exp(1j*self.k*th.sqrt(-self.z**2+xx**2+yy**2)))/(-self.z**2+xx**2+yy**2))
#         H_i =  th_ff(h_i)*self.pix_size**2


#         self.register_buffer("H", H.cfloat())
#         self.register_buffer("H_i", H_i.cfloat())

   
#     def forward(self, X):
#         """Performs forward propagation"""
#         return th_iff(th_ff(X) * self.H)
    
#     def inverse(self, X):
#         """Performs inverse propagation"""
#         return th_iff(th_ff(X) * self.H_i)
    

# class PropagatorRSTF(th.nn.Module):
#     """
   
#     Rayleigh–Sommerfeld transfer function propagation 
#     better for shorter  distances
#     Δx >= λz/L
#     """

#     def __init__(
#         self,
#         pixel_num,
#         pix_size,
#         wavelength,
#         z,
#     ):
#         super().__init__()

#         self.num = pixel_num
#         self.pix_size = pix_size
#         self.wavelength = wavelength
#         self.z = z
#         self.k = 2*th.pi/wavelength

  




#         fx,fy = freq_grid(pixel_num=self.num,pixel_size=self.pix_size)

#         H = th.exp(2j*th.pi*self.z*th.sqrt(1- (self.wavelength*fx )**2 - (self.wavelength*fy)**2)/self.wavelength)
#         H_i = th.exp(2j*th.pi*-self.z*th.sqrt(1- (self.wavelength*fx )**2 - (self.wavelength*fy)**2)/self.wavelength)


#         self.register_buffer("H", H.cfloat())
#         self.register_buffer("H_i", H_i.cfloat())

   
#     def forward(self, X):
#         """Performs forward propagation"""
#         return th_iff(th_ff(X) * self.H)
    
#     def inverse(self, X):
#         """Performs inverse propagation"""
#         return th_iff(th_ff(X) * self.H_i)
    







class PropagatorFIR(th.nn.Module):
    """
   
    Fresnel impulse responce propagation 
    better for longer distances
    Δx <= λz/L
    """

    def __init__(
        self,
        pixel_num,
        pix_size,
        wavelength,
        z,
    ):
        super().__init__()


        # k= self.k
        # lm=self.wavel
        # x,y =self.grid()
        # dx= self.cell
        # #np.exp(1j*k*z)
        # h = (1/(1j*lm*z)) * np.exp(((1j*k)/(2*z))*(x**2+y**2))
        # H = ff(h)*dx**2



        self.num = pixel_num
        self.pix_size = pix_size
        self.wavelength = wavelength
        self.z = z
        self.k = 2*th.pi/wavelength



  

        xx, yy = grid(pixel_size=self.pix_size,pixel_num=self.num)


        h = (1/(1j* self.wavelength*self.z)) * th.exp(((1j*self.k)/(2* self.z))*(xx**2+yy**2))
        H = th_ff(h)*self.pix_size**2

        h_i = (1/(1j* self.wavelength*-self.z)) * th.exp(((1j*self.k)/(2* -self.z))*(xx**2+yy**2))
        H_i = th_ff(h_i)*self.pix_size**2


        self.register_buffer("H", H.cfloat())
        self.register_buffer("H_i", H_i.cfloat())

   
    def forward(self, X):
        """Performs forward propagation"""
        return th_iff(th_ff(X) * self.H)
    
    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff(th_ff(X) * self.H_i)
    


class PropagatorFTF(th.nn.Module):
    """
   
    Fresnel transfer function
    better for shorter distances
    Δx >= λz/L
    """

    def __init__(
        self,
        pixel_num,
        pix_size,
        wavelength,
        z,
    ):
        super().__init__()

        self.num = pixel_num
        self.pix_size = pix_size
        self.wavelength = wavelength
        self.z = z
        self.k = 2*th.pi/wavelength

  

        fx,fy = freq_grid(pixel_size = self.pix_size, pixel_num=self.num)


        H = th.exp((-1j*th.pi*self.wavelength*self.z)*(fx**2 +fy**2))
        H_i = th.exp((-1j*th.pi*self.wavelength*-self.z)*(fx**2 +fy**2))


        self.register_buffer("H", H.cfloat())
        self.register_buffer("H_i", H_i.cfloat())

   
    def forward(self, X):
        """Performs forward propagation"""
        return th_iff(th_ff(X) * self.H)
    
    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff(th_ff(X) * self.H_i)