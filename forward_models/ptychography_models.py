"""
Contains models required for the AD-based ptychography
"""
# import torch.nn.functional as F
# import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader
# from propagators import grid

from .element_models.probe_models import (
    ProbeComplexShotToShotConstant,
    ProbeDoubleRealShotToShotConstant,
    ProbeComplexShotToShotVariable,
    ProbeDoubleRealShotToShotVariable,
    ProbeComplexShotToShotVariable_coherent,
    ProbeComplexShotToShotVariable_coherent_incoherent,
    ProbeComplexShotToShotConstant_variable_int,
    ProbeComplexShotToShotConstant_variable_int_supported
)

from .element_models.sample_models import (
    SampleComplex,
    SampleDoubleReal,
    SampleRefractive,
    SampleVariableThickness,
    SampleRefractiveConstrained,
    SampleRefractiveConstrained_split,
)

from .element_models.propagator_models import (
    PropagatorFraunhFluxPreserving,
    PropagatorFresnelSingleTransformFLuxPreserving,
)

from .element_models.shifter_models import (
    ShifterDefault,
    ShifterElastic,
    ShifterRotational,
    ShifterRotationalElastic,
    ShifterDefault_Fourrier,
    Shifter3DFourrieRreduced
)

from .element_models.support_models import (
    Support,
)

from .element_models.bragg_rotators import (
    Probe_3d_projector_reduced,
    Probe_3d_projector_reduced_near90,
    Probe_3d_projector_reduced_near90_cut
)

from .element_models.probe_models import (
    ProbeComplexShotToShotConstant,
    ProbeDoubleRealShotToShotConstant,
    ProbeComplexShotToShotVariable,
    ProbeDoubleRealShotToShotVariable,
    ProbeComplexShotToShotVariable_coherent,
    ProbeComplexShotToShotVariable_coherent_incoherent
)

from ..utils import Pad

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Ptychography models___________


class PtychographyModelTransmission(th.nn.Module):
    """Describes transmission ptychography experiment.
    Returns [scan_positions_num,modes_num,probe_y,probe_x] tensor,
    which later can be treated depending on the coherent and other properties of the experiment
    """

    def __init__(
        self,
        probe_type,
        sample_type,
        shifter_type,
        propagator_type,
        probe_params,
        sample_params,
        shifter_params,
        propagator_params,
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
        elif probe_type =='ProbeComplexShotToShotVariable_coherent':
            self.Probe = ProbeComplexShotToShotVariable_coherent(**probe_params)
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
        elif sample_type == 'refractive_constrained':
            self.Sample = SampleRefractiveConstrained(**sample_params)
        else:
            raise ValueError("Unknown sample_type")

        if shifter_type == "default":
            self.Shifter = ShifterDefault(**shifter_params)
        elif shifter_type == "elastic":
            self.Shifter = ShifterElastic(**shifter_params)
        elif shifter_type == "rotational":
            self.Shifter = ShifterRotational(**shifter_params)
        elif shifter_type == "rotational_elastic":
            self.Shifter = ShifterRotationalElastic(**shifter_params)
        elif shifter_type == "default_fourrier":
            self.Shifter = ShifterDefault_Fourrier(**shifter_params)
        else:
            raise ValueError("Unknown shifter_type")

        if propagator_type == "Fraunhofer":
            self.Propagator = PropagatorFraunhFluxPreserving(**propagator_params)
        elif propagator_type == "Fresnel_single_transform":
            self.Propagator = PropagatorFresnelSingleTransformFLuxPreserving(
                **propagator_params
            )
        else:
            raise ValueError("Unknown propagator_type")

    def forward(self, scan_numbers):
        """Estimate the measured diffraction patterns for corresponding scan numbers"""
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        return self.Propagator(
            self.Shifter(self.Sample(), scan_numbers)[:, None, :, :]
            * self.Probe(scan_numbers)
        )

    
    

class PtychographyModelTransmissionSupported(th.nn.Module):
    """Describes transmission ptychography experiment.
    Returns [scan_positions_num,modes_num,probe_y,probe_x] tensor,
    which later can be treated depending on the coherent and other properties of the experiment
    """

    def __init__(
        self,
        probe_type,
        sample_type,
        shifter_type,
        propagator_type,
        probe_params,
        sample_params,
        shifter_params,
        propagator_params,
        support_params
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
        elif probe_type =='ProbeComplexShotToShotVariable_coherent':
            self.Probe = ProbeComplexShotToShotVariable_coherent(**probe_params)
        elif probe_type == 'ProbeComplexShotToShotVariable_coherent_incoherent':
            self.Probe = ProbeComplexShotToShotVariable_coherent_incoherent(**probe_params)
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
        elif sample_type == 'refractive_constrained':
            self.Sample = SampleRefractiveConstrained(**sample_params)
        elif sample_type == 'SampleRefractiveConstrained_split':
            self.Sample = SampleRefractiveConstrained_split(**sample_params)
        else:
            raise ValueError("Unknown sample_type")

        if shifter_type == "default":
            self.Shifter = ShifterDefault(**shifter_params)
        elif shifter_type == "elastic":
            self.Shifter = ShifterElastic(**shifter_params)
        elif shifter_type == "rotational":
            self.Shifter = ShifterRotational(**shifter_params)
        elif shifter_type == "rotational_elastic":
            self.Shifter = ShifterRotationalElastic(**shifter_params)
        elif shifter_type == "default_fourrier":
            self.Shifter = ShifterDefault_Fourrier(**shifter_params)
        else:
            raise ValueError("Unknown shifter_type")

        if propagator_type == "Fraunhofer":
            self.Propagator = PropagatorFraunhFluxPreserving(**propagator_params)
        elif propagator_type == "Fresnel_single_transform":
            self.Propagator = PropagatorFresnelSingleTransformFLuxPreserving(
                **propagator_params
            )
        else:
            raise ValueError("Unknown propagator_type")
            
        self.Support = Support(**support_params)

    def forward(self, scan_numbers):
        """Estimate the measured diffraction patterns for corresponding scan numbers"""
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        return self.Propagator(
            self.Shifter(self.Sample(), scan_numbers)[:, None, :, :]
            * self.Support(self.Probe(scan_numbers))
        )
    





class PtychographyModelBragg(th.nn.Module):
    """Describes Bragg projection ptychography experiment.
    Approximates each individual position-rotation dependant diffraction pattern using projection formalism
    from  Hruszkewycz, Stephan O., et al. "High-resolution three-dimensional structural microscopy by single-angle Bragg ptychography." 
    Nature materials 16.2 (2017): 244-251. 

    Returns [scan_positions_num,modes_num,probe_y,probe_x] tensor,
    which later can be treated depending on the coherent and other properties of the experiment
    """

    def __init__(
        self,
        probe_type,
        sample_type,
        shifter_type,
        propagator_type,
        projector_type,
        probe_params,
        sample_params,
        shifter_params,
        propagator_params,
        support_params,
        projector_params
    ):
        """probe_type one of 'complex_shot_to_shot_constant','double_real_shot_to_shot_constant',
        'Probe_complex_shot_to_shot_variable','Probe_double_real_shot_to_shot_variable'.

        sample_type one of 'complex_TF','double_real_TF','refractive','thickness'.

        shifter_type one of 'fourrier_3d'.

        propagator_type one of 'Fraunhofer','Fresnel_single_transform'.

        projector_type one of 'Probe_3d_projector_reduced'.

        probe_params: (init_probe, number_of_positions=None, modal_weights=None)
        last two parameters for variable probe case

        sample_params: (sample_size=None, init_sample=None) one must be not None

        shifter_params: (init_shifts,init_rotations,borders,sample_size,
        max_correction,max_rotation_correction,)
        depending on the shifter type

        propagator_params: (pixel_size,pixel_num,wavelength,z,) depending on the propagator type

        projector_params : (interaction_length_pixels:int, probe_shape:th.Size, 
        bragg_angle:float,
        axis:int = -2,)

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
        elif probe_type =='ProbeComplexShotToShotVariable_coherent':
            self.Probe = ProbeComplexShotToShotVariable_coherent(**probe_params)
        elif probe_type == 'ProbeComplexShotToShotVariable_coherent_incoherent':
            self.Probe = ProbeComplexShotToShotVariable_coherent_incoherent(**probe_params)
        elif probe_type == 'ProbeComplexShotToShotConstant_variable_int':
            self.Probe = ProbeComplexShotToShotConstant_variable_int(**probe_params)
        elif probe_type == 'ProbeComplexShotToShotConstant_variable_int_supported':
            self.Probe = ProbeComplexShotToShotConstant_variable_int_supported(**probe_params)
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
        elif sample_type == 'refractive_constrained':
            self.Sample = SampleRefractiveConstrained(**sample_params)
        elif sample_type == 'SampleRefractiveConstrained_split':
            self.Sample = SampleRefractiveConstrained_split(**sample_params)
        else:
            raise ValueError("Unknown sample_type")

        if shifter_type == "Shifter3DFourrieRreduced":
            self.Shifter = Shifter3DFourrieRreduced(**shifter_params)
        else:
            raise ValueError("Unknown shifter_type")

        if propagator_type == "Fraunhofer":
            self.Propagator = PropagatorFraunhFluxPreserving(**propagator_params)
        elif propagator_type == "Fresnel_single_transform":
            self.Propagator = PropagatorFresnelSingleTransformFLuxPreserving(
                **propagator_params
            )
        else:
            raise ValueError("Unknown propagator_type")
        

        if projector_type == "Probe_3d_projector_reduced":
            self.Projector = Probe_3d_projector_reduced(**projector_params)
        elif projector_type == 'Probe_3d_projector_reduced_near90':
            self.Projector = Probe_3d_projector_reduced_near90(**projector_params)
        elif projector_type == 'Probe_3d_projector_reduced_near90_cut':
            self.Projector == Probe_3d_projector_reduced_near90_cut(**projector_params)
        else:
            raise ValueError("Unknown projector_type")
            
            
        self.Support = Support(**support_params)
        


    def forward(self, scan_numbers):
        """Estimate the measured diffraction patterns for corresponding scan numbers"""
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        # return self.Propagator(
        #     self.Shifter(self.Sample(), scan_numbers)[:, None, :, :]
        #     * self.Support(self.Probe(scan_numbers))
        # )


        return self.Propagator(Pad((self.Shifter(self.Support(self.Sample()),scan_numbers)[:,None,...]*self.Projector(self.Probe(scan_numbers)[0])[None,...]).sum(axis = 2),self.Projector.shape_for_pad))