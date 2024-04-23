"""Contains routines to controll optimization flow
"""
import time
import torch as th
import matplotlib.pyplot as plt
import numpy as np
from .loss_and_regularizers import get_regularization_estimator


class DataParallelAtributeTransparent(th.nn.DataParallel):
    """Iheritor of the DataParalel providing acces to the model attributes"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def save_reconstruction_results(
    model_to_save, optimizer, reconstruction_summary, path, name=""
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

def load_model(filename,to_cpu = True):
    pass
    





def fit(Epoch_num,
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
    """Performs reconstruction

    Args:
        Epoch_num (int): number of total epochs
        optimizer (torch.optim object): optimizer to use
        data_loader (iterable): data loader
        Measured_data (dict): measured data
        forward_model (torch.nn.Module): forward model
        report_interval (int, optional): report interval. Defaults to 1.
        early_stopping (float, optional): stoping criterion. Defaults to 1e-10.
        pre_iter_hook (function, optional): function to call before each iteration. Defaults to lambda x:None.
        post_iter_hook (function, optional): function to call after each iteration. Defaults to lambda x:None.
        forward_computation (function, optional): function to compute forward model output. Defaults to lambda x:None.
        loss_computation (function, optional): function to compute loss. Defaults to lambda x:None.

    Returns:
        dict: dictionary containing optimizer, reconstruction_summary and model
    """

    err_long =np.zeros((Epoch_num,len(data_loader)))

    num_bunch = len(data_loader)
    reconstruction_summary = {}
    reconstruction_summary["start_time"] = time.time()
    reconstruction_summary["error"] = []
    # print(err_long.shape)


    total_iter_num = 0
    for epoch in range(Epoch_num):  # global iterations
        for data_id in data_loader:

            total_iter_num += 1
            

            with th.no_grad():#apply pre-iteration hooks
                pre_iter_hook(forward_model)

            optimizer.zero_grad(set_to_none=True)

            model_output = forward_computation(forward_model)
            Measured_batch = Measured_data[data_id]


            loss = loss_computation(model_output,forward_model)
            err_long[epoch,total_iter_num%num_bunch] = loss.item()
            loss.backward()

            with th.no_grad():
                post_iter_hook(forward_model)

            optimizer.step()



            if epoch % report_interval == 0:
                print('\r',epoch, err_long[epoch].mean(), "||", end='')
                if err_long[epoch].mean() <early_stopping:
                    print('EARLY STOPPING REACHED')
                    
                    plt.figure()
                    plt.plot(reconstruction_summary["error"])
                    plt.title("Reconstruction error")
                    plt.show()

                    return {'Optimizer':optimizer,
                            'reconstruction_summary':reconstruction_summary,
                            'model':forward_model}

                    

    reconstruction_summary["end_time"] = time.time()
    reconstruction_summary["error"] = err_long

    print(f"RECONSTRUCTION IS FINISHED")
    time_of_reconstruction = reconstruction_summary["end_time"]-reconstruction_summary["start_time"]
    print(f"Total {time_of_reconstruction}|   |{time_of_reconstruction/Epoch_num} per epoch")

    plt.figure()
    plt.plot(reconstruction_summary["error"])
    plt.title("Reconstruction error")
    plt.show()

    return {'Optimizer':optimizer,
            'reconstruction_summary':reconstruction_summary,
            'model':forward_model}
