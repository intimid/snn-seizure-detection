import numpy as np
import matplotlib.pyplot as plt

import os
from load_data import get_tuh_raw

data_x, data_y = get_tuh_raw(test=True)

# Model parameters.
sample_length = 12 * 1000
n_sample = 1000
n_stdp_neurons = 38
n_rstdp_neurons = 2
n_seiz = np.count_nonzero(data_y[:n_sample])
n_non_seiz = n_sample - n_seiz

# Initialise folder and file names.
folder = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\weights"
subfolder = "14.10.2023_0"
foldername = os.path.join(folder, subfolder)

filename_stdp_spikes = "stdp_test_spikes.npy"
filename_rstdp_spikes = "rstdp_test_spikes.npy"
filename_hit_miss_rate = "hit_miss_rate_test.npy"

# Load the files.
stdp_spikes = np.load(os.path.join(foldername, filename_stdp_spikes))
rstdp_spikes = np.load(os.path.join(foldername, filename_rstdp_spikes))
hit_miss_rate = np.load(os.path.join(foldername, filename_hit_miss_rate))

stdp_spike_idxs = stdp_spikes[0]
stdp_spike_times = stdp_spikes[1]
rstdp_spike_idxs = rstdp_spikes[0]
rstdp_spike_times = rstdp_spikes[1]

# Categorise spikes by neuron and sample.
stdp_seiz_spikes_by_neuron = [0] * n_stdp_neurons
stdp_nonseiz_spikes_by_neuron = [0] * n_stdp_neurons
stdp_seiz_spikes_by_sample = [0] * n_seiz
stdp_nonseiz_spikes_by_sample = [0] * n_non_seiz

seiz_samples = np.where(data_y[:n_sample] == 1)[0]
non_seiz_samples = np.where(data_y[:n_sample] == 0)[0]

# Separate the spike times into 12-second sample windows.
for i in range(len(stdp_spike_times)):
    sample_no = int(stdp_spike_times[i] // sample_length)
    if data_y[sample_no]:
        stdp_seiz_spikes_by_neuron[int(stdp_spike_idxs[i])] += 1
        stdp_seiz_spikes_by_sample[np.where(seiz_samples == sample_no)[0][0]] += 1
    else:
        stdp_nonseiz_spikes_by_neuron[int(stdp_spike_idxs[i])] += 1
        stdp_nonseiz_spikes_by_sample[np.where(non_seiz_samples == sample_no)[0][0]] += 1

# Print the number of seizure samples in the data.
print(f"Number of seizure samples: {n_seiz} / {n_sample}")
# Print the hit and miss rates.
print(f"Hit rate: {hit_miss_rate[0][0]}   |   Miss rate: {hit_miss_rate[0][1]}")

# Plot the number of spikes per neuron in a stacked bar chart.
fig = plt.figure()
bottom = np.zeros(n_stdp_neurons)
plt.bar(np.arange(n_stdp_neurons), stdp_nonseiz_spikes_by_neuron, label="Non-Seizure", bottom=bottom)
bottom += np.array(stdp_nonseiz_spikes_by_neuron)
plt.bar(np.arange(n_stdp_neurons), stdp_seiz_spikes_by_neuron, label="Seizure", bottom=bottom)
plt.title("Number of spikes per neuron")
plt.xlabel("Neuron index")
plt.ylabel("Number of spikes")
plt.legend()

# Plot the number of spikes per sample.
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].bar(np.arange(n_non_seiz), stdp_nonseiz_spikes_by_sample)
axs[0].set_title("Non-seizure")
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Number of spikes")
axs[1].bar(np.arange(n_seiz), stdp_seiz_spikes_by_sample)
axs[1].set_title("Seizure")
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Number of spikes")

plt.show()


spike_indices = np.load('spike_indices.npy')
spike_times = np.load('spike_times.npy')

sample_length = 12 * 1000
n_sample = 500
n_neurons = 38

n_seiz = np.count_nonzero(data_y[:n_sample])
n_non_seiz = n_sample - n_seiz

# spikes_by_sample = [[[0] for _ in range(38)] for _ in range(n_sample)]
seiz_spikes_by_neuron = [0] * n_neurons
non_seiz_spikes_by_neuron = [0] * n_neurons
seiz_spikes_by_sample = [0] * n_seiz
non_seiz_spikes_by_sample = [0] * n_non_seiz

seiz_samples = np.where(data_y[:n_sample] == 1)[0]
non_seiz_samples = np.where(data_y[:n_sample] == 0)[0]

# Separate spike times into 12-second sample windows.
for i in range(len(spike_times)):
    sample_no = int(spike_times[i] // sample_length)
    # spikes_by_sample[sample_no][spike_indices[i]][0] += 1
    if data_y[sample_no]:
        seiz_spikes_by_neuron[spike_indices[i]] += 1
        seiz_spikes_by_sample[np.where(seiz_samples == sample_no)[0][0]] += 1
    else:
        non_seiz_spikes_by_neuron[spike_indices[i]] += 1
        non_seiz_spikes_by_sample[np.where(non_seiz_samples == sample_no)[0][0]] += 1

# Print the number of seizure samples in the data.
print(f"Number of seizure samples: {np.count_nonzero(data_y[:n_sample])} / {n_sample}")

# Plot the number of spikes in a stacked bar chart.
fig = plt.figure()
bottom = np.zeros(n_neurons)
plt.bar(np.arange(n_neurons), non_seiz_spikes_by_neuron, label="Non-seizure", bottom=bottom)
bottom += np.array(non_seiz_spikes_by_neuron)
plt.bar(np.arange(n_neurons), seiz_spikes_by_neuron, label="Seizure", bottom=bottom)
plt.title("Number of spikes per neuron")
plt.xlabel("Neuron index")
plt.ylabel("Number of spikes")
plt.legend()

# Plot the number of spikes in each sample.
fig, axs = plt.subplots(1, 2, sharey=True)
axs[0].bar(np.arange(n_non_seiz), non_seiz_spikes_by_sample)
axs[0].set_title("Non-seizure")
axs[0].set_xlabel("Sample index")
axs[0].set_ylabel("Number of spikes")
axs[1].bar(np.arange(n_seiz), seiz_spikes_by_sample)
axs[1].set_title("Seizure")
axs[1].set_xlabel("Sample index")
axs[1].set_ylabel("Number of spikes")

plt.show()