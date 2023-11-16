import numpy as np
import matplotlib.pyplot as plt

import os
import idx2numpy
import json
import pickle
from load_data import get_tuh_raw


def return_program_args(data_type):
    """Returns program-specific arguments based on user settings.
    """
    match data_type:
        case 'thr_encoded':
            data_subfolder = 'threshold_encoded'
        case 'stft':
            data_subfolder = 'tuh_stft_ica_devpei12s_npy'
        case 'mnist':
            data_subfolder = 'mnist'
    return data_subfolder


def get_data(mode, tuh_subfolder='threshold_encoded', remote_deploy=False):
    """Gets the data from the remote or local TUH server.
    """
    if remote_deploy:
        # Set the folder name of the remote TUH data.
        foldername = '/home/tim/SNN Seizure Detection/TUH'
        foldername = os.path.join(foldername, tuh_subfolder)
    else: # TODO: Fix for local TUH data.
        # Set the folder name of the local TUH data.
        foldername = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\data"
        foldername = os.path.join(foldername, tuh_subfolder)

    # Used for POC demonstration of model on MNIST data.
    if tuh_subfolder == 'mnist':
        match mode:
            case 'train':
                filename_x = 'train-images.idx3-ubyte'
                filename_y = 'train-labels.idx1-ubyte'
            case 'test':
                filename_x = 't10k-images.idx3-ubyte'
                filename_y = 't10k-labels.idx1-ubyte'
        data_x = idx2numpy.convert_from_file(os.path.join(foldername, filename_x))
        data_y = idx2numpy.convert_from_file(os.path.join(foldername, filename_y))
        return data_x, data_y

    # Set the file names of the data and labels.
    filename_x = f"{mode}x.npy"
    filename_y = f"{mode}y.npy"

    # Load the data and labels.
    data_x = np.load(os.path.join(foldername, filename_x), allow_pickle=True)
    data_y = np.load(os.path.join(foldername, filename_y))

    return data_x, data_y


# Initialise folder and file names.
folder = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\weights"
subfolder = "08.11.2023_1"
foldername = os.path.join(folder, subfolder)
filename_stdp_spikes = "stdp_test_spikes.pkl"
filename_rstdp_spikes = "rstdp_test_spikes.pkl"
filename_hit_miss_rate = "hit_miss_rate_test.npy"

# Load the files.
with open(os.path.join(foldername, filename_stdp_spikes), "rb") as f:
    stdp_spikes = pickle.load(f)
with open(os.path.join(foldername, filename_rstdp_spikes), "rb") as f:
    rstdp_spikes = pickle.load(f)
hit_miss_rate = np.load(os.path.join(foldername, filename_hit_miss_rate))

# Model parameters.
# Model parameters.
with open(os.path.join(foldername, "model_params.txt"), "r") as f:
    model_params = f.readlines()
model_params = [param.strip() for param in model_params]
sample_length = float(model_params[7].split(": ")[1])
n_sample = int(model_params[9].split(": ")[1])
sample_length = 0.35
for filename in os.listdir(foldername):
    if filename.endswith(".json"):
        # Open the file and read the data.
        with open(os.path.join(foldername, filename), "r") as f:
            model_config = json.load(f)
n_stdp_neurons = int(model_config['network_params']['n_stdp'])
n_rstdp_neurons = int(model_config['network_params']['n_output'])

# Load the test data.
data_type = 'thr_encoded'  # 'raw' or 'stft'
data_subfolder = return_program_args(data_type)
data_x, data_y = get_data(mode='test', tuh_subfolder=data_subfolder)
if data_type == 'mnist':
    # Get only the samples with labels 0 and 1.
    data_x = data_x[np.where(data_y < 2)]
    data_y = data_y[np.where(data_y < 2)]
n_seiz = np.count_nonzero(data_y[:n_sample])
n_non_seiz = n_sample - n_seiz

# Categorise spikes by neuron and sample.
stdp_seiz_spikes_by_neuron = np.zeros(n_stdp_neurons)
stdp_nonseiz_spikes_by_neuron = np.zeros(n_stdp_neurons)
stdp_seiz_spikes_by_sample = np.zeros(n_sample)
stdp_nonseiz_spikes_by_sample = np.zeros(n_sample)

seiz_samples = np.where(data_y[:n_sample] == 1)[0]
non_seiz_samples = np.where(data_y[:n_sample] == 0)[0]

for idx in seiz_samples:
    for i in range(len(stdp_spikes[idx][0])):
        stdp_seiz_spikes_by_neuron[int(stdp_spikes[idx][0][i])] += 1
        stdp_seiz_spikes_by_sample[idx] += 1
for idx in non_seiz_samples:
    for i in range(len(stdp_spikes[idx][0])):
        stdp_nonseiz_spikes_by_neuron[int(stdp_spikes[idx][0][i])] += 1
        stdp_nonseiz_spikes_by_sample[idx] += 1

# Print the number of seizure samples in the data.
print(f"Number of seizure samples: {n_seiz} / {n_sample}")
# Print the hit and miss rates.
print(f"Hit rate: {hit_miss_rate[0][0]}   |   Miss rate: {hit_miss_rate[0][1]}")

# Plot the number of spikes per neuron in a stacked bar chart.
fig = plt.figure()
bottom = np.zeros(n_stdp_neurons)
plt.bar(np.arange(1, n_stdp_neurons+1), stdp_nonseiz_spikes_by_neuron, label="Non-Seizure", bottom=bottom)
bottom += np.array(stdp_nonseiz_spikes_by_neuron)
plt.bar(np.arange(1, n_stdp_neurons+1), stdp_seiz_spikes_by_neuron, label="Seizure", bottom=bottom)
plt.xlim([0, n_stdp_neurons+1])
plt.title("Number of spikes per neuron")
plt.xlabel("Neuron index")
plt.ylabel("Number of spikes")
plt.legend()

# Plot the percentage of non-seizure and seizure spikes per neuron.
fig = plt.figure()
percentage_per_neuron = np.zeros(n_stdp_neurons)
for i in range(n_stdp_neurons):
    percentage_per_neuron[i] = stdp_seiz_spikes_by_neuron[i] / (stdp_seiz_spikes_by_neuron[i] + stdp_nonseiz_spikes_by_neuron[i])
plt.bar(np.arange(1, n_stdp_neurons+1), percentage_per_neuron)
plt.xlim([0, n_stdp_neurons+1])
plt.title("Percentage of seizure spikes per neuron")
plt.xlabel("Neuron index")
plt.ylabel("Percentage of seizure spikes")

# Plot the number of spikes per sample.
# TODO: THIS IS BROKEN. NEED TO CHANGE THIS STACKED BAR GRAPH TO R-STDP.
fig, axs = plt.subplots(1, 2, sharey=True)
bottom = np.zeros(len(non_seiz_samples))
axs[0].bar(np.arange(len(non_seiz_samples)), stdp_nonseiz_spikes_by_sample[non_seiz_samples], 
                     label="Non-Seizure", bottom=bottom)
bottom += np.array(stdp_nonseiz_spikes_by_sample[non_seiz_samples])
axs[0].bar(np.arange(len(non_seiz_samples)), stdp_seiz_spikes_by_sample[non_seiz_samples], 
                     label="Seizure", bottom=bottom)
plt.show()
axs[0].set_title("Non-seizure")
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Number of spikes")
axs[1].bar(np.arange(n_seiz), stdp_seiz_spikes_by_sample)
axs[1].set_title("Seizure")
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Number of spikes")

plt.show()