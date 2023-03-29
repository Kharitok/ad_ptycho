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
    freq_grid,
)


class PropagatorRayleighSommerfeldTF_constant (th.nn.Module):
    """
    Rayleigh–Sommerfeld transfer function responce propagation  
    better for shorter  distances
    Δx >= λz/L
    """
    pass
    def __init__(
        self,
        pixel_size,
        pixel_num,
        wavelength,
        z,
    ):
        super().__init__()

        lm = wavelength
        fx,fy = freq_grid(pixel_size=pixel_size,pixel_num=pixel_num)
        fx.cdouble()
        fy.cdouble()
        H_ = th.exp((2j*th.pi*z_gs/lm)*th.sqrt(1- (lm*fx_ )**2 - (lm*fy_)**2)) #th.exp(2j*th.pi*z*th.sqrt(1- (lm*fx )**2 - (lm*fy)**2)/lm)
        H_inverse_ = th.exp((-1*2j*th.pi*z_gs/lm)*th.sqrt(1- (lm*fx_ )**2 - (lm*fy_)**2))
        self.register_buffer("H", H_.cfloat())
        self.register_buffer("H_inverse", H_inverse_.cfloat())


    def forward(self, X):
        """Performs forward propagation"""
        return th_iff(th_ff(X)*self.H)

    def inverse(self, X):
        """Performs inverse propagation"""
        return th_iff(th_ff(X)*self.H_inverse)




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
    

        

