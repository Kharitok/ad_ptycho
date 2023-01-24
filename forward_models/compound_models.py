"""Describes forward models (ptychography + noise, detector, etc.) for ad-based ptychography"""
# import torch.nn as nn
from typing import Tuple
import torch as th

from .ptychography_models import PtychographyModelTransmission
from .noise_models import AdditiveGaussianNoise,AdditiveGaussianNoiseVariable


class TransmissionPtychographyWithGaussianNoise(th.nn.Module):
    """Integrates noise and experiment model for transmissive prychoraphy with gaussian noise
    """

    def __init__(self, ptychography_parameters: dict, noise_parameters: dict) -> None:
        super().__init__()
        self.ptychography_model = PtychographyModelTransmission(
            **ptychography_parameters
        )
        self.noise_model = AdditiveGaussianNoise(**noise_parameters)

    def forward(self, scan_numbers) -> Tuple:
        """Estimate the measured diffraction patterns for corresponding scan numbers,
        considering noise, detector, etc.

        """
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        return (
            th.sqrt(th.sum(th.abs(self.ptychography_model(scan_numbers)) ** 2, axis=1)),
            self.noise_model.get_gaussian(),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.ptychography_model, name)
            except AttributeError:
                return getattr(self.noise_model, name)



class TransmissionPtychographyWithVariableGaussianNoise(th.nn.Module):
    """Integrates noise and experiment model for transmissive prychoraphy with gaussian noise
    """

    def __init__(self, ptychography_parameters: dict, noise_parameters: dict) -> None:
        super().__init__()
        self.ptychography_model = PtychographyModelTransmission(
            **ptychography_parameters
        )
        self.noise_model = AdditiveGaussianNoiseVariable(**noise_parameters)

    def forward(self, scan_numbers) -> Tuple:
        """Estimate the measured diffraction patterns for corresponding scan numbers,
        considering noise, detector, etc.

        """
        #         print("Sample", self.Sample().shape)
        #         print("Probe", self.Probe(scan_numbers).shape)
        #         print("scan_numbers", scan_numbers.shape)
        return (
            th.sqrt(th.sum(th.abs(self.ptychography_model(scan_numbers)) ** 2, axis=1)),
            self.noise_model.get_gaussian(),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            try:
                return getattr(self.ptychography_model, name)
            except AttributeError:
                return getattr(self.noise_model, name)