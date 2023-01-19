"""Contains routines to controll optimization flow
"""
import time
import torch as th
import matplotlib.pyplot as plt
import numpy as np


class DataParallelAtributeTransparent(th.nn.DataParallel):
    """Iheritor of the DataParalel providing acces to the model attributes"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def save_reconstruction_results(
    model_to_save, optimizer, reconstruction_summary, name, path
):
    """saves model and reconstruction data to the file"""

    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    savepath = f"{path}/{name}_{current_time}.ptym"

    th.save(
        {
            "model": model_to_save,
            "optimizer": optimizer,
            "reconstruction summary": reconstruction_summary,
        },
        savepath,
    )




def train(model,noise_generator,optimizer,data_loader,epoch_num,report_interval,backup_interval,early_stopping,Measured_data,savepath):
    reconstruction_summary ={}
    reconstruction_summary['start_time'] = time.time()
    reconstruction_summary['error'] = []



    for epoch in range(epoch_num):
        # iterate through the dataloader to process next minibatch
        for data in data_loader:
            # zero gradients
            optimizer.zero_grad(set_to_none=True)

            # estimate diffraction with forward model
            model_output =  th.sqrt(th.sum(th.abs(model(data))**2,axis=1))
            noise = noise_generator.get_gaussian()#
            Measured_batch = Measured_data[data, ...]


            # calculate loss
        loss = (
            Model_err(Approx=model_output+noise, Measured=Measured_batch, Mask=mask, mode="PNL")
            +l1_norm_reg(model_output*(1.0-mask),a1=2e-2,a2=0)
            # +l1_norm_refractive_reg(Reconstruction_model_gpu.Sample.sample,1,0)
            # +L1(Reconstruction_model_gpu.Sample.sample,a1=0,a2=1e-1)
            # +total_variation_reg(Reconstruction_model_gpu.Probe.probe,1e-3,0)
        #    +total_variation_refractive_reg(Reconstruction_model_gpu.Sample.sample,1e-1,5e-8)
        )

        # backward and optimizer step
        loss.backward()
        optimizer.step()

        # save minibatch error
        reconstruction_summary["error"].append((loss.item()))


        # print current stat
        if epoch % report_interval == 0:
            print('\r',epoch, reconstruction_summary["error"][-1], "||", end='')
            if reconstruction_summary["error"][-1] <early_stopping:
                print('EARLY STOPPING REACHED')
                break

        if (epoch + 1) % backup_interval == 0:
            backup_name = (
                "some_path"
                + "Ptychography model"
                + str(datetime.datetime.now())
                + "_"
                + str(np.around(reconstruction_summary["error"][-1], 5))
                + ".pth"
            )
            th.save(
                {
                    "model": P,
                    "optimizer": optimizer,
                    "reconstruction_summary": reconstruction_summary,
                },
                backup_name,
            )

    reconstruction_summary["end time"] = time.time()
    time_of_reconstruction = (
        reconstruction_summary["end time"] - reconstruction_summary["start_time"]
    )
    # Summarize
    print(f"RECONSTRUCTION IS FINISHED")
    print(
        f"Total {time_of_reconstruction}|   |{time_of_reconstruction/epoch_num} per epoch"
    )
    plt.figure()
    plt.plot(reconstruction_summary["error"])
    plt.title("Reconstruction error")
    plt.show()

    return (model,optimizer,reconstruction_summary)

# Save final results

# save_name = (
#     "some_path"
#     + "Ptychography model FINAL"
#     + str(datetime.datetime.now())
#     + "_"
#     + str(np.around(reconstruction_summary["error"][-1], 5))
#     + ".pth"
# )
# th.save(
#     {
#         "model": P,
#         "optimizer": optimizer,
#         "reconstruction_summary": reconstruction_summary,
#     },
#     save_name,
# )