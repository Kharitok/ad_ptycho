# import torch.nn.functional as F
# import torch.nn as nn
import torch as th
# import torch.fft as th_fft
from torch.utils.data import Dataset

# from propagators import grid

# th.pi = th.acos(th.zeros(1)).item() * 2  # which is 3.1415927410125732
# th.backends.cudnn.benchmark = True


class ShufflingScanPositionsDataset(Dataset):
    def __init__(self, scan_numbers):
        self.len = len(scan_numbers)
        self.data = th.from_numpy(scan_numbers).long().cuda()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def cpu(self):
        self.data = self.data.cpu()

    def gpu(self):
        self.data = self.data.cuda()
