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
    PropagatorFresnelSingleTransformFLuxPreserving,
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
        fx =fx.cdouble()
        fy =fy.cdouble()
        H_ = th.exp((2j*th.pi*z/lm)*th.sqrt(1- (lm*fx )**2 - (lm*fy)**2)) #th.exp(2j*th.pi*z*th.sqrt(1- (lm*fx )**2 - (lm*fy)**2)/lm)
        H_inverse_ = th.exp((-1*2j*th.pi*z/lm)*th.sqrt(1- (lm*fx )**2 - (lm*fy)**2))
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
        self.num_tilts = self.init_alphas.shape[0]
    def forward(self, X):
        """Tilts the incoming wavefield of shape [num_fields,H,W]"""
        transfer_function =  th.exp(
            1j * self.k * (self.xx[None,:,] * th.cos(self.thetas[:,None,None])+ self.yy[None,:,] * th.sin(self.thetas[:,None,None])) * th.tan(self.alphas)[:,None,None]
        )
        return X.expand(self.num_tilts,-1,-1,-1)*transfer_function[:,None,:,:]
    

        
class SingleShotPtychographyModel(th.nn.Module):
    """Describes single-shot diffraction-grating-based ptychography experiment.
    Returns [modes_num,probe_y,probe_x] tensor,
    which later can be treated depending on the coherent and other properties of the experiment
    """

    def __init__(
        self,
        probe_type,
        sample_type,
        probe_params,
        sample_params,
        propagator_params_grating_sample,
        propagator_params_sample_detector,
        tilt_params,
    ):
        """probe_type one of 'complex_shot_to_shot_constant','double_real_shot_to_shot_constant',
        'Probe_complex_shot_to_shot_variable','Probe_double_real_shot_to_shot_variable'.

        sample_type one of 'complex_TF','double_real_TF','refractive','thickness'.

        shifter_type one of 'default','elastic','rotational','rotational_elastic'.

        propagator_type one of 'Fraunhofer','Fresnel_single_transform'.

        probe_params: (init_probe, number_of_positions=None, modal_weights=None)
        last two parameters for variable probe case

        sample_params: (sample_size=None, init_sample=None) one must be not None

        shifter_params: (init_shifts,init_rotations,borders,sample_size,
        max_correction,max_rotation_correction,)
        depending on the shifter type

        propagator_params: (pixel_size,pixel_num,wavelength,z,) depending on the propagator type

        """
        super().__init__()

        if probe_type == "Probe_complex_shot_to_shot_constant":
            self.Probe = ProbeComplexShotToShotConstant(**probe_params)
        elif probe_type == "Probe_double_real_shot_to_shot_constant":
            self.Probe = ProbeDoubleRealShotToShotConstant(**probe_params)
        elif probe_type == "Probe_complex_shot_to_shot_variable":
            self.Probe = ProbeComplexShotToShotVariable(**probe_params)
        elif probe_type == "Probe_double_real_shot_to_shot_variable":
            self.Probe = ProbeDoubleRealShotToShotVariable(**probe_params)
        else:
            raise ValueError("Unknown probe_type")

        if sample_type == "complex_TF":
            self.Sample = SampleComplex(**sample_params)
        elif sample_type == "double_real_TF":
            self.Sample = SampleDoubleReal(**sample_params)
        elif sample_type == "refractive":
            self.Sample = SampleRefractive(**sample_params)
        elif sample_type == "thickness":
            self.Sample = SampleVariableThickness(**sample_params)
        else:
            raise ValueError("Unknown sample_type")


        self.Propagator_grating_sample = PropagatorRayleighSommerfeldTF_constant(**propagator_params_grating_sample)
        self.Propagator_sample_detector = PropagatorFresnelSingleTransformFLuxPreserving(**propagator_params_sample_detector)
        self.Tilt = TiltSmallAngle(**tilt_params) 

    def forward(self, ):
        """Estimate the measured diffraction patterns for corresponding scan numbers"""
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        modulated_probe_diffraction =self.Propagator_sample_detector( self.Propagator_grating_sample(self.Tilt(self.Probe()).sum(axis=0)) *self.Sample()[None,...] )
        probe_diffraction = self.Propagator_sample_detector( self.Propagator_grating_sample(self.Tilt(self.Probe()).sum(axis=0))  )
        return (modulated_probe_diffraction,probe_diffraction)

# class SingleShotPtychographyModel(th.nn.Module):
#     """Describes single-shot diffraction-grating-based ptychography experiment.
#     Returns [modes_num,probe_y,probe_x] tensor,
#     which later can be treated depending on the coherent and other properties of the experiment
#     """

#     def __init__(
#         self,
#         probe_type,
#         sample_type,
#         shifter_type,
#         propagator_type,
#         probe_params,
#         sample_params,
#         shifter_params,
#         propagator_params,
#     ):
#         """probe_type one of 'complex_shot_to_shot_constant','double_real_shot_to_shot_constant',
#         'Probe_complex_shot_to_shot_variable','Probe_double_real_shot_to_shot_variable'.

#         sample_type one of 'complex_TF','double_real_TF','refractive','thickness'.

#         shifter_type one of 'default','elastic','rotational','rotational_elastic'.

#         propagator_type one of 'Fraunhofer','Fresnel_single_transform'.

#         probe_params: (init_probe, number_of_positions=None, modal_weights=None)
#         last two parameters for variable probe case

#         sample_params: (sample_size=None, init_sample=None) one must be not None

#         shifter_params: (init_shifts,init_rotations,borders,sample_size,
#         max_correction,max_rotation_correction,)
#         depending on the shifter type

#         propagator_params: (pixel_size,pixel_num,wavelength,z,) depending on the propagator type

#         """
#         super().__init__()

#         if probe_type == "Probe_complex_shot_to_shot_constant":
#             self.Probe = ProbeComplexShotToShotConstant(**probe_params)
#         elif probe_type == "Probe_double_real_shot_to_shot_constant":
#             self.Probe = ProbeDoubleRealShotToShotConstant(**probe_params)
#         elif probe_type == "Probe_complex_shot_to_shot_variable":
#             self.Probe = ProbeComplexShotToShotVariable(**probe_params)
#         elif probe_type == "Probe_double_real_shot_to_shot_variable":
#             self.Probe = ProbeDoubleRealShotToShotVariable(**probe_params)
#         else:
#             raise ValueError("Unknown probe_type")

#         if sample_type == "complex_TF":
#             self.Sample = SampleComplex(**sample_params)
#         elif sample_type == "double_real_TF":
#             self.Sample = SampleDoubleReal(**sample_params)
#         elif sample_type == "refractive":
#             self.Sample = SampleRefractive(**sample_params)
#         elif sample_type == "thickness":
#             self.Sample = SampleVariableThickness(**sample_params)
#         else:
#             raise ValueError("Unknown sample_type")

#         if shifter_type == "default":
#             self.Shifter = ShifterDefault(**shifter_params)
#         elif shifter_type == "elastic":
#             self.Shifter = ShifterElastic(**shifter_params)
#         elif shifter_type == "rotational":
#             self.Shifter = ShifterRotational(**shifter_params)
#         elif shifter_type == "rotational_elastic":
#             self.Shifter = ShifterRotationalElastic(**shifter_params)
#         else:
#             raise ValueError("Unknown shifter_type")

#         if propagator_type == "Fraunhofer":
#             self.Propagator = PropagatorFraunhFluxPreserving(**propagator_params)
#         elif propagator_type == "Fresnel_single_transform":
#             self.Propagator = PropagatorFresnelSingleTransformFLuxPreserving(
#                 **propagator_params
#             )
#         else:
#             raise ValueError("Unknown propagator_type")

#     def forward(self, scan_numbers):
#         """Estimate the measured diffraction patterns for corresponding scan numbers"""
#         #         print("Sample", self.Sample().shape)
#         #         print("Probe", self.Probe(scan_numbers).shape)
#         #         print("scan_numbers", scan_numbers.shape)
#         return self.Propagator(
#             self.Shifter(self.Sample(), scan_numbers)[:, None, :, :]
#             * self.Probe(scan_numbers)
#         )