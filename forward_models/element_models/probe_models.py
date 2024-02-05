"""
Constains probe models for ad based ptychography
"""
# import torch.nn.functional as Fw
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# from propagators import grid

th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
th.backends.cudnn.benchmark = True


# ___________Probe models___________
class ProbeComplexShotToShotConstant_variable_int(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with complex tensor.
    """

    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        self.scaling = nn.Parameter(th.ones(init_probe.shape[0]))

    def forward(self, scan_numbers=None):
        """Returns probe function at scan_numbers positions"""
        return self.scaling[None,:,None,None]*self.probe[None, ...]
    

class ProbeComplexShotToShotConstant_variable_int_supported(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with complex tensor.
    """

    def __init__(self, init_probe,support):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        self.register_buffer('support', th.from_numpy(support).data)
        self.scaling = nn.Parameter(th.ones(init_probe.shape[0]))

    def forward(self, scan_numbers=None):
        """Returns probe function at scan_numbers positions"""
        return self.scaling[None,:,None,None]*self.probe[None, ...]*self.support[None,...]

class ProbeComplexShotToShotConstant(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with complex tensor.
    """

    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

    def forward(self, scan_numbers=None):
        """Returns probe function at scan_numbers positions"""
        return self.probe[None, ...]


class ProbeDoubleRealShotToShotConstant(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe
    constant from one shot to another. Implemented with two real tensors.
    """

    def __init__(self, init_probe):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe_real = nn.Parameter(
                (th.from_numpy(init_probe.real).float())[None, :, :]
            )
            self.probe_imag = nn.Parameter(
                (th.from_numpy(init_probe.imag).float())[None, :, :]
            )
        else:
            self.probe_real = nn.Parameter((th.from_numpy(init_probe.real).float()))
            self.probe_imag = nn.Parameter((th.from_numpy(init_probe.imag).float()))

    def forward(self, scan_numbers=None):
        """Returns probe function at scan_numbers positions"""
        return th.complex(self.probe_real, self.probe_imag)[None, ...]


class ProbeComplexShotToShotVariable(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe with  shot to shot unique
    modal weights. Implemented with complex tensor.
    """

    def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        if modal_weights is not None:
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif number_of_positions is not None:
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return self.probe[None, :, :] * self.modal_weights[scan_numbers, :, None, None]

    
class ProbeComplexShotToShotVariable_coherent(ProbeComplexShotToShotVariable):
    """Orthogonal probe relaxation
    """

#     def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
#         super().__init__()
#         if len(init_probe.shape) == 2:
#             self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
#         else:
#             self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

#         if modal_weights is not None:
#             self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
#         elif number_of_positions is not None:
#             self.modal_weights = nn.Parameter(
#                 (th.ones((number_of_positions, self.probe.shape[0])).float())
#             )
#         else:
#             raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return (self.probe[None, :, :] * self.modal_weights[scan_numbers, :, None, None]).sum(axis=1)[:,None,:,:]
    


class ProbeComplexShotToShotVariable_coherent_incoherent(th.nn.Module):
    """Orthogonal probe relaxation
    """

    def __init__(self, init_probe, number_of_positions=None,decoher_modes=None, modal_weights = None,  ):
        super().__init__()
        # raise ValueError("NOT IMPLEMENTED")
        if len(init_probe.shape) == 2:
            self.probe = nn.Parameter((th.from_numpy(init_probe).cfloat())[None, :, :])
        else:
            self.probe = nn.Parameter(th.from_numpy(init_probe).cfloat())

        if modal_weights is not None:
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif (number_of_positions is not None) and (decoher_modes is not None):
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, decoher_modes,self.probe.shape[0])).float())
            )
        else:
            raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return (self.modal_weights[scan_numbers,...,None,None]*self.probe[None,None, :, :]).sum(axis=2)  #(self.probe[None, :, :] * self.modal_weights[scan_numbers, :, None, None]).sum(axis=1)[:,None,:,:]   
    



class ProbeDoubleRealShotToShotVariable(th.nn.Module):
    """Multymodal (can be usefd for fully coherent with one mode) probe with  shot to shot unique
    modal weights. Implemented with two real tensors.
    """

    def __init__(self, init_probe, number_of_positions=None, modal_weights=None):
        super().__init__()
        if len(init_probe.shape) == 2:
            self.probe_real = nn.Parameter(
                (th.from_numpy(init_probe.real).float())[None, :, :]
            )
            self.probe_imag = nn.Parameter(
                (th.from_numpy(init_probe.imag).float())[None, :, :]
            )
        else:
            self.probe_real = nn.Parameter((th.from_numpy(init_probe.real).float()))
            self.probe_imag = nn.Parameter((th.from_numpy(init_probe.imag).float()))

        if modal_weights is not None:
            self.modal_weights = nn.Parameter(th.from_numpy(modal_weights).float())
        elif number_of_positions is not None:
            self.modal_weights = nn.Parameter(
                (th.ones((number_of_positions, self.probe.shape[0])).float())
            )
        else:
            raise ValueError("Either number_of_positions or modal_weights should be given")

    def forward(self, scan_numbers):
        """Returns probe function at scan_numbers positions"""
        return (
            th.complex(self.probe_real, self.probe_imag)[None, :, :]
            * self.modal_weights[scan_numbers, :, None, None]
        )
