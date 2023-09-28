import numpy as np
import os

def get_TUH_npy_data(filename, mmap_mode=None):
    """Gets the TUH .npy data that has been processed by Luis.

    mmap_mode is passed to np.load() to allow for memory mapping for 
    particularly large files.
    """
    data_dir = "/home/tim/SNN Seizure Detection/TUH/reshuffle"

    return np.load(os.path.join(data_dir, filename), mmap_mode=mmap_mode)

def slice_data(data, slice_pct):
    num_samples = data.shape[0]
    num_samples_slice = int(num_samples * slice_pct)
    sliced_data = data[:num_samples_slice]

    return sliced_data


if __name__ == '__main__':
    # Get the training data.
    data = get_TUH_npy_data("trainx.npy", mmap_mode='r')
    # Slice 10% of the data.
    sliced_data = slice_data(data, slice_pct=0.1)

    # Save the sliced data in a new directory.
    save_dir = "/home/tim/SNN Seizure Detection/TUH/sliced"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "trainx.npy"), sliced_data)

    # Get the label data.
    data = get_TUH_npy_data("trainy.npy", mmap_mode='r')
    # Slice 10% of the data.
    sliced_data = slice_data(data, slice_pct=0.1)

    # Save the sliced data in the same directory.
    np.save(os.path.join(save_dir, "trainy.npy"), sliced_data)