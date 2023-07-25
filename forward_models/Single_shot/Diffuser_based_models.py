import torch as th
import torch.nn as nn
import numpy as np
import torch.fft as th_fft


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



class Diffuser_based_CDI(th.nn.Module):
    """
    A forward model desctibing the propagation of the wavefield throught the randomized diffuser,
    attemting to do the single-shot reconstructions
    """

    def __init__(self,diffuser_TF,propagator_params_sample_detector,sample_type,sample_params):
        
        super().__init__()
        self.register_buffer("D_TF", diffuser_TF)
        self.Propagator_diffuser_detector = PropagatorFresnelSingleTransformFLuxPreserving(**propagator_params_sample_detector)
        
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

    def forward(self,):
        """Performs forward propagation"""
        return self.Propagator_diffuser_detector( self.Sample()*self.D_TF)
