import torch as th
import torch.nn as nn

from ..element_models.probe_models import (
    ProbeComplexShotToShotConstant,
    ProbeDoubleRealShotToShotConstant,
    ProbeComplexShotToShotVariable,
    ProbeDoubleRealShotToShotVariable,
)

from ..element_models.sample_models import (
    SampleComplex,
    SampleDoubleReal,
    SampleRefractive,
    SampleVariableThickness,
)

from ..element_models.propagator_models import (
    grid,
    th_ff,
    th_iff,
)


class PropagatorRayleighSommerfeldTF (th.nn.Module):
    """
    Rayleigh–Sommerfeld transfer function responce propagation  
    better for shorter  distances
    Δx >= λz/L
    """
    pass
    # def __init__(
    #     self,
    #     pixel_size,
    #     pixel_num,
    #     wavelength,
    #     z,
    # ):
    #     super().__init__()

    #     lm = wavelength
    #     k = 2 * th.pi / lm

    #     x1, y1 = grid(pixel_size, pixel_num)

    #     x2, y2 = fourrier_scaled_grid(lm, z, pixel_size, pixel_num)

    #     mul1 = th.exp((1j * k / (2 * z)) * (x2**2 + y2**2))
    #     mul2 = th.exp((1j * k / (2 * z)) * (x1**2 + y1**2))

    #     mul1_inv = th.exp((1j * k / (2 * (-z)) * (x1**2 + y1**2)))
    #     mul2_inv = th.exp((1j * k / (2 * (-z)) * (x2**2 + y2**2)))

    #     self.register_buffer("mul1", mul1.cfloat())
    #     self.register_buffer("mul2", mul2.cfloat())

    #     self.register_buffer("mul1_inv", mul1_inv.cfloat())
    #     self.register_buffer("mul2_inv", mul2_inv.cfloat())

    #     self.num = pixel_num

    # def forward(self, X):
    #     """Performs forward propagation"""
    #     return self.mul1 * th_ff(X * self.mul2) / self.num

    # def inverse(self, X):
    #     """Performs inverse propagation"""
    #     return self.mul1_inv * th_iff(X * self.mul2_inv) * self.num




class TiltSmallAngle (th.nn.Module):
    """
    Differentiable tilt operator

        Applies  phase mask accourding: T(x y) = exp (jk *(x cos theta  y sin theta)
        alpha - tilt angle
        theta - rotation angle (x axis 0) ^| -90, _| 90
    """
    pass

    def __init__(self,
                pixel_size,
                pixel_num,
                wavelength,
                alphas,
                thetas,
                 ) -> None:
        
        super().__init__()

        self.wavelength = wavelength
        self.k = 2 * th.pi / wavelength
        self.pixel_size = pixel_size
        self.pixel_num = pixel_num
        self.init_thetas = thetas
        self.init_alphas = alphas
        self.num_angles = len(alphas)

        xx_,yy_ = grid(pixel_size, pixel_num) 

        self.register_buffer("xx", xx_.cfloat())
        self.register_buffer("yy", yy_.cfloat())

        self.alphas = nn.Parameter((th.from_numpy(self.init_alphas.real).float()))
        self.thetas = nn.Parameter((th.from_numpy(self.init_thetas.real).float()))

    def forward(self, X):
        """Tilts the incoming wavefield of shape [num_fields,H,W]"""
        transfer_function =  th.exp(
            1j * self.k * (self.xx[None,:,] * th.cos(tilt.thetas[:,None,None])+ self.yy[None,:,] * th.sin(self.thetas[:,None,None])) * th.tan(self.alphas)[:,None,None]
        )
        return X*transfer_function[:,None,:,:]
    

        

