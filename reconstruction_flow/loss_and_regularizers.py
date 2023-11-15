"""
Contains loss and regularization functions required for the AD-based ptychography
"""
# import torch.nn.functional as F
# import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Regularization staff___________
def l1_norm_reg(input_tensor: th.tensor, a1: float = 1, a2: float = 1) -> th.Tensor:
    """Calculates L1 norm of a tensor X.
    a1,a2 are weights for the modulus and phase respectively."""

    return (1 / th.numel(input_tensor)) * (
        a1 * th.abs(th.abs(input_tensor)) + a2 * th.abs(th.angle(input_tensor))
    ).sum()


def l1_norm_refractive_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates L1 norm of a tensor X representing complex tensor in refractive form.
    phase = Re(X), modulus = exp(-Im(x))
    a1,a2 are weights for the modulus and phase respectively."""

    trans_l1_sum = (th.exp(-1 * th.imag(input_tensor))).sum()
    phase_l1_sum = (th.abs(th.real(input_tensor))).sum()
    return (1 / th.numel(input_tensor)) * ((a1 * trans_l1_sum) + (a2 * phase_l1_sum))


def total_variation_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates total variantion of the input tensor weighted with a1 for modulus and a2 for phase
    """
    abs_x = th.abs(input_tensor)
    ang_x = th.angle(input_tensor)

    return (1 / th.numel(input_tensor)) * a1 * (
        th.sum(th.abs(abs_x[..., :, 1:] - abs_x[..., :, :-1]))
        + th.sum(th.abs(abs_x[..., 1:, :] - abs_x[..., :-1, :]))
    ) + a2 * (
        th.sum(th.abs(ang_x[..., :, 1:] - ang_x[..., :, :-1]))
        + th.sum(th.abs(ang_x[..., 1:, :] - ang_x[..., :-1, :]))
    )


def total_variation_refractive_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates total variantion of the input tensor in refractive form ,
    weighted with a1 for modulus and a2 for phase"""
    abs_x = th.imag(input_tensor)
    ang_x = th.real(input_tensor)

    return (1 / th.numel(input_tensor)) * a1 * (
        th.sum(th.abs(th.diff(abs_x, dim=-1))) + th.sum(th.abs(th.diff(abs_x, dim=-2)))
    ) + a2 * (
        th.sum(th.abs(th.diff(ang_x, dim=-1))) + th.sum(th.abs(th.diff(ang_x, dim=-2)))
    )


def total_variation_refractive_isotropic_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates total variantion of the input tensor in refractive form ,
    weighted with a1 for modulus and a2 for phase"""
    abs_x = th.imag(input_tensor)
    ang_x = th.real(input_tensor)

    return (1 / th.numel(input_tensor)) * a1 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(abs_x, dim=-1)[..., :-1, :], 2)
                + th.pow(th.diff(abs_x, dim=-2)[..., :-1], 2)
                + 1e-10
            )
        )
    ) + a2 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(ang_x, dim=-1)[..., :-1, :], 2)
                + th.pow(th.diff(ang_x, dim=-2)[..., :-1], 2)
                + 1e-10
            )
        )
    )


def total_variation_second_refractive_isotropic_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates total variantion of the input tensor in refractive form ,
    weighted with a1 for modulus and a2 for phase"""
    abs_x = th.imag(input_tensor)
    ang_x = th.real(input_tensor)

    return (1 / th.numel(input_tensor)) * a1 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(abs_x, n=2, dim=-1)[..., :-2, :], 2)
                + th.pow(th.diff(abs_x, n=2, dim=-2)[..., :-2], 2)
                + 1e-10
            )
        )
    ) + a2 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(ang_x, n=2, dim=-1)[..., :-2, :], 2)
                + th.pow(th.diff(ang_x, n=2, dim=-2)[..., :-2], 2)
                + 1e-10
            )
        )
    )




def generalized_total_variation_second_refractive_isotropic_reg(
    input_tensor: th.tensor, a1: float = 1, a2: float = 1
) -> th.Tensor:
    """Calculates total variantion of the input tensor in refractive form ,
    weighted with a1 for modulus and a2 for phase"""
    abs_x = th.imag(input_tensor)
    ang_x = th.real(input_tensor)

    return (1 / th.numel(input_tensor)) * a1 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(abs_x, n=2, dim=-1)[..., :-2, :], 2)
                + th.pow(th.diff(abs_x, n=2, dim=-2)[..., :-2], 2)
                + 1e-10
            )
        )
    ) + a2 * (
        th.sum(
            th.sqrt(
                th.pow(th.diff(ang_x, n=2, dim=-1)[..., :-2, :], 2)
                + th.pow(th.diff(ang_x, n=2, dim=-2)[..., :-2], 2)
                + 1e-10
            )
        )
    )


# ___________Loss Criterion___________


class LossEstimator:
    """Class containing different loss functions to be used during the optimization
    consult with [1] Noise models for low counting rate coherent diffraction imaging 
    https://doi.org/10.1364/OE.20.025914
    [2] Maximum-likelihood refinement for coherent diffractive imaging
    [3] Maximum-likelihood estimation in ptychography in the presence of Poisson-Gaussian noise statistics

    for LSQ it's better to use sqrt(I) rather than I
    For PNL  I
    Counting is copied from [1] and expects I 
    Pu_Ga is copied from [3] with sigma_masked being variance of the readout noise estimated from darks apriory and  masked accourdingly

    """

    def __init__(self, Mask=None):
        if Mask is not None:
            self.Mask = Mask
        else:
            self.Mask = None

        self.LSQ = th.nn.MSELoss()
        #         self.LSQ_rel = torch.nn.MSELoss(reduction="sum")
        self.PNL_usual = th.nn.PoissonNLLLoss(log_input=False)
        self.PNL_log = th.nn.PoissonNLLLoss(log_input=True)

    def __call__(self, Approx, Measured, mode="LSQ", Mask=None,sigma_masked=None):
        if not (Mask is None):
            Approx = Approx * Mask
            Measured = Measured * Mask
        elif not(self.mask is None):
            Approx = Approx * self.Mask
            Measured = Measured * self.Mask

        if mode == "LSQ":
            return self.LSQ(Approx, Measured)
        elif mode == "LSQ_rel":
            return (
                ((Measured - Approx) ** 2).sum(dim=[1, 2])
                / ((Measured**2).sum(dim=[1, 2]))
            ).mean()
        elif mode == "PNL":
            return self.PNL_usual(Approx, Measured)
        elif mode == "PNL_log":
            return self.PNL_log(Approx, Measured)
        elif mode == "Counting":
            return (((Measured-Approx)/(th.sqrt(Measured)+1e-7))**2).mean()
        elif mode == "Pu_Ga":
            return (th.log(Approx+sigma_masked+1e-7)+ ((Measured-Approx)**2)/(Approx+sigma_masked+1e-7)).mean()
        else:
            raise ValueError("Unknown mode")




            +l1_norm_reg(model_output*(1.0-mask),a1=2e-2,a2=0)
            +l1_norm_refractive_reg(Reconstruction_model_gpu.Sample.sample,1e0,0)#1e-1
#             +l1_norm_reg(Reconstruction_model_gpu.Sample.sample,a1=1,a2=1e-1)
#             +total_variation_reg(th.abs(Reconstruction_model_gpu.Probe.probe)**2,1e-5,)
#            +total_variation_refractive_isotropic_reg(Reconstruction_model_gpu.Sample.sample,0,1e-7)
        #    +total_variation_second_refractive_isotropic_reg(Reconstruction_model_gpu.Sample.sample,1e-1,5e-7)


def get_regularization_estimator(regularization_parameters: dict) -> callable:
    """creates regularization estimator for use in auto reconstructions"""

    def regularization(model, model_output, mask):
        res = (
            l1_norm_reg(
                model_output * (1.0 - mask), **regularization_parameters["l1_outside"]
            )
            + l1_norm_refractive_reg(
                model.Sample.sample, **regularization_parameters["l1_sample"]
            )
            + total_variation_reg(
                th.abs(model.Probe.probe) ** 2, **regularization_parameters["tv_probe"]
            )
            + total_variation_second_refractive_isotropic_reg(
                model.Sample.sample, **regularization_parameters["tv_sample"]
            )
        )
        return res

    return regularization
