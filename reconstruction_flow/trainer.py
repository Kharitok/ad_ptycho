import time
import torch as th
import matplotlib.pyplot as plt
import numpy as np
from .loss_and_regularizers import get_regularization_estimator, LossEstimator
from ..data_preprocessing.preprocessing import (
    save_dataset_for_reconstruction,
    read_dataset_for_reconstruction,
)


# model,
# optimizer,
# data_loader,
# epoch_num,
# report_interval,
# backup_interval,
# early_stopping,
# measured_data,
# mismatch_estimator,
# mask,
# savepath,
# regularization_params,
# name=''


class ReconstructionManager:
    def __init__(self, modelpath, datasetpath, batch_size):
        self.model, self.optimizer = self.load_model_optimizer(modelpath)
        self.load_dataset(datasetpath)
        self.loss_estimator = LossEstimator()
        self.batch_size = batch_size

        self.dataset = ShufflingScanPositionsDataset(np.arange(diffr_modulus.shape[0]))
        self.dataset.cpu()
        self.random_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        self.mask = th.from_numpy(self.bad_pixels_ROI.astype(np.float32)).cuda()

        self.Measured_data = th.from_numpy(diffr_modulus.astype(np.float32)).cuda()

        self.model = model.cuda()
        self.Measured_data = th.from_numpy(self.diffr_modulus.astype(np.float32))
        self.Measured_data = self.Measured_data.cuda()
        self.dataset.data = self.dataset.data.cuda()

    def train(self):
        pass

    def save(self):
        pass

    def load_model_optimizer(self, load_path):
        loaded = th.load(load_path)
        return (loaded["model"], loaded["optimizer"])

    def load_dataset(self, dataset_path):
        loaded_data = read_dataset_for_reconstruction(dataset_path)
        self.detector_pixel_size = loaded_data["detector_pixel_size"]
        self.diffr_modulus = loaded_data["diffr_modulus"]
        self.distance_sample_detector = loaded_data["distance_sample_detector"]
        self.max_sample_size = loaded_data["max_sample_size"]
        self.pi_x_centered_pix = loaded_data["pi_x_centered_pix"]
        self.pi_y_centered_pix = loaded_data["pi_y_centered_pix"]
        self.probe_interp_norm = loaded_data["probe_interp_norm"]
        self.rec_resolution = loaded_data["rec_resolution"]
        self.wavelength = loaded_data["wavelength"]
        self.bad_pixels_ROI = loaded_data["mask"]
        self.eta = loaded_data["eta"]

    def set_optimizer_params(self,new_params):
        self.optimizer.param_groups[0]['lr'] = new_params['probe']#1e-1# PROBE
        self.optimizer.param_groups[1]['lr'] = new_params['scan positions']#5e-3# SCAN POSITIONS
        self.optimizer.param_groups[2]['lr'] = new_params['sample']#1e-1# SAMPLE
        self.optimizer.param_groups[3]['lr'] = new_params['noise']# NOISE 

    def set_regularization_params(self):
        pass
