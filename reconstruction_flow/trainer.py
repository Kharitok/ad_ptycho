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



def train(Epoch_num,
          optimizer,
          data_loader,
          Measured_data,
          forward_model,
          report_interval =1,
          early_stopping = 1e-10,
          pre_iter_hook= lambda x:None,
          post_iter_hook= lambda x:None,
          forward_computation = lambda x:None,
          loss_computation = lambda x:None,

          ):

    err_long =np.zeros((Epoch_num,len(data_loader)))

    num_bunch = len(data_loader)
    reconstruction_summary = {}
    reconstruction_summary["start_time"] = time.time()
    reconstruction_summary["error"] = []


    total_iter_num = 0
    for epoch in range(Epoch_num):  # global iterations
        for data_id in data_loader:

            total_iter_num += 1
            

            with th.no_grad:#apply pre-iteration hooks
                pre_iter_hook(forward_model)

            optimizer.zero_grad(set_to_none=True)

            model_output = forward_computation(forward_model)
            Measured_batch = Measured_data[data_id]


            loss = loss_computation(model_output,forward_model)
            err_long[epoch,total_iter_num//num_bunch] = loss.item()
            loss.backward()

            with th.no_grad():
                post_iter_hook(forward_model)

            optimizer.step()



            if epoch % report_interval == 0:
                print('\r',epoch, reconstruction_summary["error"][-1], "||", end='')
                if err_long[-1] <early_stopping:
                    print('EARLY STOPPING REACHED')
                    break

    reconstruction_summary["end_time"] = time.time()
    reconstruction_summary["error"] = err_long

    print(f"RECONSTRUCTION IS FINISHED")
    print(f"Total {time_of_reconstruction}|   |{time_of_reconstruction/Epoch_num} per epoch")

    plt.figure()
    plt.plot(reconstruction_summary["error"])
    plt.title("Reconstruction error")
    plt.show()

    return {'Optimizer':optimizer,
            'reconstruction_summary':reconstruction_summary,
            'model':forward_model}



# err_long= None

# Epoch_num = 10000
# report_interval = 1
# backup_interval = 10000
# early_stopping = -1e20
# stagnaniton_toll=1e-1
# last_stuck=0
# to_sw = False
# if err_long is None:
#     err_long = []
#     err_hist = []


# reconstruction_summary = {}
# reconstruction_summary["start_time"] = time.time()
# reconstruction_summary["error"] = []

# for epoch in range(Epoch_num):  # global iterations
    
#         with th.no_grad():
#             # probe_g.probe *= support[None,...]
#             probe_guess.probe *= sup_f[None,...]
            
#             pass

#         # zero gradients
#         optimizer.zero_grad(set_to_none=True)

#         # estimate diffraction with forward model

#         # model_output =  th.sqrt( th.sum(th.abs( prop_ff(modulator*prop_nf(probe_g([0])) ))**2,axis=1))[0]
#         # model_output =  th.sqrt( th.sum(th.abs( prop_ff(modulator*ifftnd_t(probe_g([0]),(-1,-2)) ))**2,axis=1))[0]
#         # model_output = ( th.abs(prop_ff(ifftnd_t(probe_guess.probe[0],(-1,-2))*modulator)))**2
#         model_output = ( th.abs(prop_ff(ifftnd_t(probe_guess.probe[0],(-1,-2))*modulator)))
#         # model_output =  th.sum(th.abs( prop_ff(modulator*prop_nf(ifftnd_t(probe_g([0]),(-1,-2)) )))**2,axis=1)[0]
#         # model_output =  th.sum(th.abs( prop_ff(modulator*ifftnd_t(probe_g([0]),(-1,-2)) ))**2,axis=1)[0]
# #         noise = additive_noise.get_gaussian()#
#         Measured_batch = Measured_data[0]

#         # calculate loss
#         loss = ( 
#             # Model_err(Approx=model_output, Measured=Measured_batch,  mode="LSQ",Mask = mask_det)
#             L_diffr(Measured =Measured_batch, Approx=model_output,mask =mask_det )
#             +1e-3*tv_2_reg(th.abs(probe_guess.probe[0]))
#             +1e-4*tv_2_reg(th.abs(ifftnd_t(probe_guess.probe[0],(-1,-2))))
#             +1e-3*l1_norm_reg(th.abs(ifftnd_t(probe_guess.probe[0],(-1,-2))))
#             # +1e-9*((~support_est)*th.abs(ifftnd_t(probe_guess.probe,(-1,-2)))).sum()
#             # +th.abs((~sup_f)*probe_guess.probe.detach()).sum()
#             # +1e-2*((~support_est)*th.abs(ifftnd_t(probe_g([0]),(-1,-2)))).sum()
#             # +1e-2*l1_norm_reg(th.abs(ifftnd_t(probe_guess.probe,(-1,-2)))**2)
#             # +1e-3*(th.abs(ifftnd_t(probe_guess.probe,(-1,-2)))*(~support_est)).sum()
#             # +3e-1*tv_2_reg(th.abs(ifftnd_t(probe_g([0]),(-1,-2))))
#             # +1e-4*((~support_2)*th.abs(prop_nf(probe_g([0])))).sum()
#             # +1e0*l1_norm_reg(th.abs(probe_g.probe))

#         )


#         # maybe calculate some alternative metrics without gradients
#         with th.no_grad():
#             pass
#             # loss_rel = Model_err(
#             #     Approx=output, Measured=Measured_batch, Mask=None, mode="LSQ_rel"
#             # )
#             # err_rel.append(float(loss_rel.cpu()))

#         # backward and optimizer step
#         loss.backward()
#         with th.no_grad():
#             # probe_g.probe.grad = probe_g.probe.grad*support[None,...]
#             probe_guess.probe.grad = probe_guess.probe.grad*sup_f[None,...]
#             pass
            
#         optimizer.step()

#         # save minibatch error
#         reconstruction_summary["error"].append((loss.item()))


#         # print current stat
#         if epoch % report_interval == 0:
#             print('\r',epoch, reconstruction_summary["error"][-1], "||", end='')
#             if reconstruction_summary["error"][-1] <early_stopping:
#                 print('EARLY STOPPING REACHED')
#                 break
            
#             if to_sw and ( epoch - last_stuck) >100 and  (np.abs(reconstruction_summary["error"][-1]-np.mean(reconstruction_summary["error"][-20:-1]))<=stagnaniton_toll*np.mean(reconstruction_summary["error"][-20:-1]))and(epoch>last_stuck+100):
#                 print('we stuck')
#                 support = sw_sup(6,0.18)
#                 last_stuck = epoch

#                 l_probe  =5e-1#<=6e-3
#                 l_modal_weights  =0#2e2

#                 optimizer = th.optim.Adam([
#                                         {'params': probe_g.probe,'lr':l_probe, 'weight_decay':0},# 
#                                         {'params': probe_g.modal_weights,'lr':l_modal_weights,'weight_decay':0},#

#                                     ],weight_decay=0,lr=0)




# # At the end of the reconstruction
# reconstruction_summary["end time"] = time.time()
# time_of_reconstruction = (
#     reconstruction_summary["end time"] - reconstruction_summary["start_time"]
# )
# # Summarize
# print(f"RECONSTRUCTION IS FINISHED")
# print(
#     f"Total {time_of_reconstruction}|   |{time_of_reconstruction/Epoch_num} per epoch"
# )
# plt.figure()
# plt.plot(reconstruction_summary["error"])
# plt.title("Reconstruction error")
# plt.show()




with th.no_grad():
    plt.figure()
    plt.imshow(th.abs(probe_guess.probe[0]).detach().cpu())
    plt.colorbar()
    
    
    plt.figure()
    plt.imshow(th.abs(probe_correct.probe[0]).detach().cpu())
    plt.colorbar()
    
    
    plt.figure()
    plt.imshow(np.abs(ifftnd_t(probe_correct.probe,(-1,-2)).detach().cpu().numpy()[0]))
    plt.colorbar()
    
    plt.figure()
    plt.imshow(np.abs(ifftnd_t(probe_guess.probe,(-1,-2)).detach().cpu().numpy()[0]))
    plt.colorbar()
    

    

    
    plt.figure()
    plt.imshow(np.abs(constr_probe.numpy().copy()))
    plt.colorbar()
    
    
    