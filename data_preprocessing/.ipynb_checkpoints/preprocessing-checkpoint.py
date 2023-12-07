"""Implementation of the routines for  ptychography data preprocessing """

import h5py
import numpy as np


def save_dataset_for_reconstruction(
    data_to_save: dict, filename: str, filepath: str = "./", owerwrite: bool = False
) -> None:
    """Saves dataset ready to run a recnstruction from"""
    to_save = None
    try:
        with h5py.File(f"{filepath}{filename}.h5", "r"):
            if owerwrite:
                to_save = True

            else:
                raise ValueError(
                    f"{filepath}{filename}.h5  is already existing, use with"
                    " owerwrite=True to owerwrite."
                )

    except (IOError, OSError):
        to_save = True

    if to_save is not None:
        with h5py.File(f"{filepath}{filename}.h5", "w") as file_to_save:
            for key in data_to_save:
                if not np.isscalar(data_to_save[key]):
                    file_to_save.create_dataset(
                        f"{key}",
                        data=data_to_save[key],
                        compression="gzip",
                        compression_opts=4,
                    )
                else:
                    file_to_save.create_dataset(f"{key}", data=data_to_save[key])


def read_dataset_for_reconstruction(filepath: str):
    """reads dataset from the file"""
    try:
        with h5py.File(f"{filepath}", "r") as file_to_load:
            loaded_dict = {}
            for key in file_to_load:
                loaded_dataset = file_to_load.get(key)
                if loaded_dataset.size == 1:
                    loaded_dict[key] = loaded_dataset[()]
                else:
                    loaded_dict[key] = loaded_dataset[:]
            return loaded_dict

    except (IOError, OSError):
        raise ValueError(f"{filepath} is not existing")
