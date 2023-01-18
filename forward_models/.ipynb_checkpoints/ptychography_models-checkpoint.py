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
    Probe_complex_shot_to_shot_constant,
    Probe_double_real_shot_to_shot_constant,
    Probe_complex_shot_to_shot_variable,
    Probe_double_real_shot_to_shot_variable,
)

from .element_models.propagator_models import (
    Propagator_Fresnel_single_transform_flux_preserving,
    Propagator_Fraunh_intensity_flux_preserving,

)

from .element_models.sample_models import (
    Sample_complex_TF,
    Sample_double_real_TF,
    Sample_refractive,
    Sample_thickness,
)

from .element_models.shifter_models import (
    ShifterDefault,
    Shifter_elastic,
    Shifter_rotational,
    Shifter_rotational_elastic,
)




# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Ptychography models___________


class Ptychography_model_transmission(th.nn.Module):
    """Describes transmission ptychography experiment. Returns [scan_positions_num,modes_num,probe_y,probe_x] tensor,
    which later can be treated depending on the coherent and other properties of the experiment"""

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

        shifter_params: (init_shifts,init_rotations,borders,sample_size,max_correction,max_rotation_correction,)
        depending on the shifter type

        propagator_params: (pixel_size,pixel_num,wavelength,z,) depending on the propagator type

        """
        super().__init__()

        if probe_type == "Probe_complex_shot_to_shot_constant":
            self.Probe = Probe_complex_shot_to_shot_constant(**probe_params)
        elif probe_type == "Probe_double_real_shot_to_shot_constant":
            self.Probe = Probe_double_real_shot_to_shot_constant(**probe_params)
        elif probe_type == "Probe_complex_shot_to_shot_variable":
            self.Probe = Probe_complex_shot_to_shot_variable(**probe_params)
        elif probe_type == "Probe_double_real_shot_to_shot_variable":
            self.Probe = Probe_double_real_shot_to_shot_variable(**probe_params)
        else:
            raise ValueError("Unknown probe_type")

        if sample_type == "complex_TF":
            self.Sample = Sample_complex_TF(**sample_params)
        elif sample_type == "double_real_TF":
            self.Sample = Sample_double_real_TF(**sample_params)
        elif sample_type == "refractive":
            self.Sample = Sample_refractive(**sample_params)
        elif sample_type == "thickness":
            self.Sample = Sample_thickness(**sample_params)
        else:
            raise ValueError("Unknown sample_type")

        if shifter_type == "default":
            self.Shifter = ShifterDefault(**shifter_params)
        elif shifter_type == "elastic":
            self.Shifter = Shifter_elastic(**shifter_params)
        elif shifter_type == "rotational":
            self.Shifter = Shifter_rotational(**shifter_params)
        elif shifter_type == "rotational_elastic":
            self.Shifter = Shifter_rotational_elastic(**shifter_params)
        else:
            raise ValueError("Unknown shifter_type")

        if propagator_type == "Fraunhofer":
            self.Propagator = Propagator_Fraunh_intensity_flux_preserving(
                **propagator_params
            )
        elif propagator_type == "Fresnel_single_transform":
            self.Propagator = Propagator_Fresnel_single_transform_flux_preserving(
                **propagator_params
            )
        else:
            raise ValueError("Unknown propagator_type")

    def forward(self, scan_numbers):
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        return self.Propagator(
            self.Shifter(self.Sample(), scan_numbers)[:, None, :, :]
            * self.Probe(scan_numbers)
        )
