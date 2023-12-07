"""
Support models for AD-based ptychography
"""
# import torch.nn.functional as F
import torch.nn as nn
import torch as th

# import torch.fft as th_fft
# from torch.utils.data import Dataset, DataLoader

# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


# ___________Support models___________


class Support (th.nn.Module):
    """

    """
    pass
    def __init__(
    self,support
    ):
        super().__init__()

        self.register_buffer("support", support)

    def forward(self, X):
        """Performs forward propagation"""
        return self.support[None,...]*X