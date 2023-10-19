import numpy as np
import matplotlib.pyplot as plt

import os

# Model parameters.
save_fq = 2

# Initialise folder and file names.
folder = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\weights"
subfolder = "19.10.2023_0"
foldername = os.path.join(folder, subfolder)

filename_w_stdp = "w_stdp.npy"
filename_w_inh = "w_inh.npy"
filename_w_rstdp = "w_rstdp.npy"
filename_w_stdp_final = "w_stdp_final.npy"
filename_w_inh_final = "w_inh_final.npy"
filename_w_rstdp_final = "w_rstdp_final.npy"
filename_hit_miss_rate = "hit_miss_rate.npy"

# Load the weights.
w_stdp = np.load(os.path.join(foldername, filename_w_stdp))
w_inh = np.load(os.path.join(foldername, filename_w_inh))
w_rstdp = np.load(os.path.join(foldername, filename_w_rstdp))
w_stdp_final = np.load(os.path.join(foldername, filename_w_stdp_final))
w_inh_final = np.load(os.path.join(foldername, filename_w_inh_final))
w_rstdp_final = np.load(os.path.join(foldername, filename_w_rstdp_final))

# Flatten the first two dimensions of the aggregated weights.
w_stdp = np.reshape(w_stdp, (-1, w_stdp.shape[-1]))
w_inh = np.reshape(w_inh, (-1, w_inh.shape[-1]))
w_rstdp = np.reshape(w_rstdp, (-1, w_rstdp.shape[-1]))
# Append the final weights to the aggregated weights.additional_weights
w_stdp = np.vstack((w_stdp, w_stdp_final[np.newaxis, :]))
w_inh = np.vstack((w_inh, w_inh_final[np.newaxis, :]))
w_rstdp = np.vstack((w_rstdp, w_rstdp_final[np.newaxis, :]))

# Plot the initial and final weights.
fig, axs = plt.subplots(3, 1)
axs[0].plot(w_stdp[0])
axs[0].plot(w_stdp[-1])
axs[0].set_title("STDP weights")
axs[0].set_ylabel("w")
axs[0].legend(["Initial", "Final"])
axs[1].plot(w_inh[0])
axs[1].plot(w_inh[-1])
axs[1].set_title("Inhibitory weights")
axs[1].set_ylabel("w")
axs[1].legend(["Initial", "Final"])
axs[2].plot(w_rstdp[0])
axs[2].plot(w_rstdp[-1])
axs[2].set_title("R-STDP weights")
axs[2].set_ylabel("w")
axs[2].set_xlabel("Neuron index")
axs[2].legend(["Initial", "Final"])
plt.tight_layout()

# Plot the change in weights over time.
fig, axs = plt.subplots(3, 1)
sample_numbers = np.arange(len(w_stdp)) * save_fq
w_stdp_changes = [sum(abs(w_stdp[i+1] - w_stdp[i])) for i in range(len(w_stdp) - 1)]
axs[0].plot(sample_numbers[1:], w_stdp_changes)
axs[0].set_title("STDP weights")
axs[0].set_ylabel("Total weight change")
w_inh_changes = [sum(abs(w_inh[i+1] - w_inh[i])) for i in range(len(w_inh) - 1)]
axs[1].plot(sample_numbers[1:], w_inh_changes)
axs[1].set_title("Inhibitory weights")
axs[1].set_ylabel("Total weight change")
r_stdp_changes = [sum(abs(w_rstdp[i+1] - w_rstdp[i])) for i in range(len(w_rstdp) - 1)]
axs[2].plot(sample_numbers[1:], r_stdp_changes)
axs[2].set_title("R-STDP weights")
axs[2].set_ylabel("Total weight change")
axs[2].set_xlabel("Sample number")
plt.tight_layout()

# Plot the hit and miss rates over each epoch.
hit_miss_rate = np.load(os.path.join(foldername, filename_hit_miss_rate))
plt.figure()
plt.plot(np.arange(len(hit_miss_rate[0])), hit_miss_rate[0])
plt.plot(np.arange(len(hit_miss_rate[1])), hit_miss_rate[1])
plt.title("Hit and miss rates")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.legend(["Hit rate", "Miss rate"])
plt.tight_layout()

plt.show()