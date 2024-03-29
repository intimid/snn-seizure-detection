import librosa
import numpy as np
import os

from load_data import get_tuh_data, preprocess_tuh_raw
from neural_eeg_encoder import DataEncoder


def downsample_data(data:np.ndarray, fs:int, fs_new:int):
    """Downsample the EEG data from fs to fs_new.

    Args:
        data (np.ndarray): The EEG data.
        fs (int): The original sampling frequency.
        fs_new (int): The new sampling frequency.

    Returns:
        np.ndarray: The downsampled EEG data.
    """
    return librosa.resample(data, orig_sr=fs, target_sr=fs_new)


if __name__ == '__main__':
    # Get the TUH data.
    mode='train'
    data, labels = get_tuh_data(get_remote=True, tuh_subfolder='reshuffle', mode=mode, mmap_mode=None)
    n_samples = len(data)


    # # Downsample the data.
    # fs = 250
    # fs_new = 100
    # data_downsampled = np.empty((n_samples, 19, 1200))
    # for i in range(n_samples):
    #     if i % 100 == 0:
    #         print(f"Downsampling data {i} of {n_samples}...")
    #     for j in range(19):
    #         data_downsampled[i][j] = downsample_data(data[i][j], fs, fs_new)


    # Remove 60 Hz mains noise.
    data = preprocess_tuh_raw(data)

    # Print the min, max, standard deviation, and mean of the data.
    print(f"Min: {np.min(data)}")
    print(f"Max: {np.max(data)}")
    print(f"Standard deviation: {np.std(data)}")
    print(f"Mean: {np.mean(data)}")
    # Print the percentiles of the data.
    print(f"Percentiles: {np.percentile(data, [0, 25, 50, 75, 100])}")
    exit()

    # Encode the data.
    encoder = DataEncoder(num_thresholds=19, n_channels=19, fs=250, 
                          data_range=[-200, 200], outlier_thresholds=[-800, 800])
    data_encoded = np.array([[None, None] for _ in range(n_samples)], dtype=object)
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"Encoding data {i} of {n_samples}...")
        data_encoded[i] = encoder.threshold_neuron_encode_multichannel(data[i])

    save_data = True
    save_remote = True
    tuh_subfolder = 'threshold_encoded'
    if save_data:
        if save_remote:
            # Set the folder name of the remote TUH data.
            foldername = '/home/tim/SNN Seizure Detection/TUH'
            save_dir = os.path.join(foldername, tuh_subfolder)
        else: # TODO: Fix for local TUH data.
            # Set the folder name of the local TUH data.
            save_dir = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data\\tuh_threshold_encoded"

        # Set the file names of the data and labels.
        filename_x = f"{mode}x.npy"
        filename_y = f"{mode}y.npy"

        # Save the encoded data.
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(os.path.join(save_dir, filename_x), data_encoded)
        np.save(os.path.join(save_dir, filename_y), labels)

        print(f"\nSuccessfully saved {mode} data to {save_dir}.")


    # folder = "/mnt/data4_datasets/yikai_file/tuh_data_preprocess/tuh_stft_ica_devpei12s"
    # _, _, files = next(os.walk(folder))
    # file_count = len(files)
    # print(file_count)

    # # Get the size of this folder.
    # folder_size = 0
    # for root, dirs, files in os.walk(folder):
    #     for file in files:
    #         folder_size += os.path.getsize(os.path.join(root, file))
    # # Convert to GB.
    # folder_size = folder_size / 1e9
    # print(f"Folder size: {folder_size} GB")