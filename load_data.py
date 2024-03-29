import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from scipy import signal
import matplotlib.pyplot as plt

import random
import os
import re


def get_local_tuh_dev(file_count=None, sz_ratio=None, mmap_mode=None):
    foldername = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data\\tuh_stft_ica_devpei12s_npy"
    data, labels = get_random_files_local_tuh_dev(foldername, file_count, mmap_mode)
    if sz_ratio != None:
        data, labels = force_seizure_ratio(data, labels, sz_ratio)
    data = transpose_data(data)
    data = unsqueeze_data(data)
    print(f"Data shape: {data.shape}")

    # Cast data and labels to torch tensors.
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    return data, labels


def get_random_files_local_tuh_dev(foldername, file_count=None, mmap_mode=None):
    sz_regex = re.compile(r'_seiz_')

    # Initialise lists to hold data and labels.
    if file_count == None:
        file_count = len(os.listdir(foldername))
    data = np.empty([file_count, 19, 125, 23])
    labels = np.empty(file_count)

    # Randomly sample files from dataset folder.
    if file_count == None:
        filenames = os.listdir(foldername)
    else:
        filenames = random.sample(os.listdir(foldername), file_count)
    for i, filename in enumerate(filenames):
        # Add data to list.
        filepath = os.path.join(foldername, filename)
        data[i] = np.load(filepath, mmap_mode=mmap_mode)
        # Add label to list.
        if sz_regex.search(filename) != None:
            labels[i] = 1
        else:
            labels[i] = 0

    return data, labels


def force_seizure_ratio(data, labels, sz_ratio):
    # Calculate how many seizure samples need to be added/removed.
    req_sz_count = int(len(labels) * sz_ratio)
    current_sz_count = np.count_nonzero(labels)
    sz_count_diff = req_sz_count - current_sz_count

    if sz_count_diff > 0:
        # Randomly mark non-seizure samples to be replaced.
        non_sz_indices = np.where(labels == 0)[0]
        remove_indices = random.sample(list(non_sz_indices), sz_count_diff)
        # Randomly sample seizure samples to replace non-seizure samples.
        sz_indices = np.where(labels == 1)[0]
        if sz_count_diff > len(sz_indices):
            quo, rem = divmod(sz_count_diff, len(sz_indices))
            add_indices = list(sz_indices)*quo + random.sample(list(sz_indices), rem)
        else:
            add_indices = random.sample(list(sz_indices), sz_count_diff)
        # Replace non-seizure samples with seizure samples.
        data[remove_indices] = data[add_indices]
        labels[remove_indices] = labels[add_indices]
    elif sz_count_diff < 0:
        # Randomly mark seizure samples to be replaced.
        sz_indices = np.where(labels == 1)[0]
        remove_indices = random.sample(list(sz_indices), abs(sz_count_diff))
        # Randomly sample non-seizure samples to replace seizure samples.
        non_sz_indices = np.where(labels == 0)[0]
        if abs(sz_count_diff) > len(non_sz_indices):
            quo, rem = divmod(abs(sz_count_diff), len(non_sz_indices))
            add_indices = list(non_sz_indices)*quo + random.sample(list(non_sz_indices), rem)
        else:
            add_indices = random.sample(list(non_sz_indices), abs(sz_count_diff))
        # Replace seizure samples with non-seizure samples.
        data[remove_indices] = data[add_indices]
        labels[remove_indices] = labels[add_indices]

    return data, labels


def transpose_data(data):
    data = np.transpose(data, (0, 3, 1, 2))
    return data


def unsqueeze_data(data):
    data = np.expand_dims(data, axis=2)
    return data


def split_train_test(data, label, ratio):
    train_x = data[:int(len(data) * ratio)]
    train_y = label[:int(len(label) * ratio)]
    test_x = data[int(len(data) * ratio):]
    test_y = label[int(len(label) * ratio):]
    return train_x, train_y, test_x, test_y


def prelim_analyse_data(data):
    # Calculate the min, max, standard deviation, and mean of the data.
    print(f"Min: {torch.min(data)}")
    print(f"Max: {torch.max(data)}")
    print(f"Standard deviation: {torch.std(data)}")
    print(f"Mean: {torch.mean(data)}")


def preprocess_tuh_raw(data, clip=[]):
    # Remove 60 Hz mains noise.
    b1, a1 = signal.iirnotch(w0=60, Q=30, fs=250)
    # Remove frequencies below 0.5 Hz to reduce low-frequency drift and noise.
    normal_cutoff = 0.5 / (250 / 2)
    b2, a2 = signal.butter(N=4, Wn=normal_cutoff, btype='highpass')

    n_samples = len(data)
    filtered_data = []
    for i in range(n_samples):
        if i % 100 == 0:
            print(f"Preprocessing data {i} of {n_samples}...")
        filtered_signal = signal.filtfilt(b1, a1, data[i])
        filtered_signal = signal.filtfilt(b2, a2, filtered_signal)

        if clip:
            filtered_signal = np.clip(filtered_signal, clip[0], clip[1])

        filtered_data.append(filtered_signal)

    return np.array(filtered_data)


def get_tuh_raw(test=False, sz_ratio=None, mmap_mode=None):
    foldername = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data\\tuh_raw"
    if test:
        filename_x = "testx.npy"
        filename_y = "testy.npy"
    else:
        filename_x = "trainx.npy"
        filename_y = "trainy.npy"

    data = np.load(os.path.join(foldername, filename_x), mmap_mode=mmap_mode)
    labels = np.load(os.path.join(foldername, filename_y), mmap_mode=mmap_mode)

    # Swap the second and third axes.
    # From (num_samples, voltage, channels) to (num_samples, channels, voltage).
    data = np.swapaxes(data, 1, 2)

    # Cast data and labels to torch tensors.
    data = torch.from_numpy(data).float()
    labels = torch.from_numpy(labels).float()

    return data, labels


def get_tuh_data(get_remote=True, tuh_subfolder='reshuffle', mode='train', mmap_mode=None):
    """Fetches the TUH dataset from the remote or local TUH server.
    """
    if get_remote:
        # Set the folder name of the remote TUH data.
        foldername = '/home/tim/SNN Seizure Detection/TUH'
        foldername = os.path.join(foldername, tuh_subfolder)
    else: # TODO: Fix for local TUH data.
        # Set the folder name of the local TUH data.
        foldername = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data\\tuh_raw"

    # Set the file names of the data and labels.
    filename_x = f"{mode}x.npy"
    filename_y = f"{mode}y.npy"

    # Load the data and labels.
    data = np.load(os.path.join(foldername, filename_x), mmap_mode=mmap_mode, allow_pickle=True)
    labels = np.load(os.path.join(foldername, filename_y), mmap_mode=mmap_mode)

    if tuh_subfolder == 'reshuffle':
        # Swap the second and third axes so that the data shape becomes:
        # (num_samples, channels, voltage)
        data = np.swapaxes(data, 1, 2)

    return data, labels



if __name__ == '__main__':
    foldername = "D:\\Uni\Yessir, its a Thesis\\SNN Seizure Detection\\data\\tuh_raw"
    filename_x = "trainx.npy"
    filename_y = "trainy.npy"

    # Load the data.
    data = np.load(os.path.join(foldername, filename_x))
    labels = np.load(os.path.join(foldername, filename_y))

    # Get the number of seizure samples.
    sz_count = np.count_nonzero(labels[:1000])
    print(f"Seizure count: {sz_count}")