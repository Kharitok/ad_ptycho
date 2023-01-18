"""Contains routines to controll optimization flow
"""
import time
import torch as th


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
