import numpy as np
import matplotlib.pyplot as plt

import os

# Initialise folder and file names.
folder = "D:\\Uni\\Yessir, its a Thesis\\SNN Seizure Detection\\weights"
subfolder = "08.11.2023_2"
foldername = os.path.join(folder, subfolder)

filename_w_stdp = "w_stdp.npy"
filename_w_inh = "w_inh.npy"
filename_w_rstdp = "w_rstdp.npy"
filename_w_stdp_final = "w_stdp_final.npy"
filename_w_inh_final = "w_inh_final.npy"
filename_w_rstdp_final = "w_rstdp_final.npy"
filename_hit_miss_rate = "hit_miss_rate_train.npy"

# Model parameters.
with open(os.path.join(foldername, "model_params.txt"), "r") as f:
    model_params = f.readlines()
model_params = [param.strip() for param in model_params]
epochs = int(model_params[0].split(": ")[1])
lr_init = float(model_params[1].split(": ")[1])
lr_final = float(model_params[2].split(": ")[1])
lr_decay = float(model_params[3].split(": ")[1])
save_rate = int(model_params[4].split(": ")[1])
data_type = model_params[5].split(": ")[1]
n_sample = int(model_params[6].split(": ")[1])
sample_length = float(model_params[7].split(": ")[1])

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

# Plot the positive and negative change in weights over time.
fig, axs = plt.subplots(3, 1)
save_fq = n_sample / (save_rate + 1)
save_idxs = np.ceil([i * save_fq for i in range(save_rate + 2)]).astype(int).tolist()
sample_numbers = save_idxs * epochs
epoch_numbers = [i for i in range(epochs) for _ in range((save_rate + 2))]
sample_numbers = [sample_numbers[i] + (n_sample * epoch_no) for i, epoch_no in zip(range(len(sample_numbers)), epoch_numbers)]

# Plot the change in weights over time. Ignore the weights at the end of each 
# except the last. This is to avoid double-up of weights.
sample_numbers = [item for idx, item in enumerate(sample_numbers) if idx % 6 != 0]
w_stdp_changes = [item for idx, item in enumerate(w_stdp) if idx % 6 != 0]
w_stdp_changes = [(w_stdp_changes[i+1] - w_stdp_changes[i]) for i in range(len(w_stdp_changes) - 1)]
w_stdp_changes_pos = [np.sum(w_stdp_changes[i][w_stdp_changes[i] > 0]) for i in range(len(w_stdp_changes))]
w_stdp_changes_neg = [np.sum(w_stdp_changes[i][w_stdp_changes[i] < 0]) for i in range(len(w_stdp_changes))]
axs[0].plot(sample_numbers[1:], w_stdp_changes_pos)
axs[0].plot(sample_numbers[1:], w_stdp_changes_neg)
axs[0].legend(["Positive", "Negative"])
axs[0].plot([sample_numbers[1], sample_numbers[-1]], [0, 0], '--k')
axs[0].set_title("STDP weights")
axs[0].set_ylabel("Total weight change")
w_inh_changes = [item for idx, item in enumerate(w_inh) if idx % 6 != 0]
w_inh_changes = [sum(abs(w_inh_changes[i+1] - w_inh_changes[i])) for i in range(len(w_inh_changes) - 1)]
w_inh_changes_pos = [np.sum(w_inh_changes[i][w_inh_changes[i] > 0]) for i in range(len(w_inh_changes))]
w_inh_changes_neg = [np.sum(w_inh_changes[i][w_inh_changes[i] < 0]) for i in range(len(w_inh_changes))]
axs[1].plot(sample_numbers[1:], w_inh_changes_pos)
axs[1].plot(sample_numbers[1:], w_inh_changes_neg)
axs[1].legend(["Positive", "Negative"])
axs[1].plot([sample_numbers[1], sample_numbers[-1]], [0, 0], '--k')
axs[1].set_title("Inhibitory weights")
axs[1].set_ylabel("Total weight change")
w_rstdp_changes = [item for idx, item in enumerate(w_rstdp) if idx % 6 != 0]
w_rstdp_changes = [sum(abs(w_rstdp_changes[i+1] - w_rstdp_changes[i])) for i in range(len(w_rstdp_changes) - 1)]
w_rstdp_changes_pos = [np.sum(w_rstdp_changes[i][w_rstdp_changes[i] > 0]) for i in range(len(w_rstdp_changes))]
w_rstdp_changes_neg = [np.sum(w_rstdp_changes[i][w_rstdp_changes[i] < 0]) for i in range(len(w_rstdp_changes))]
axs[2].plot(sample_numbers[1:], w_rstdp_changes_pos)
axs[2].plot(sample_numbers[1:], w_rstdp_changes_neg)
axs[2].set_title("R-STDP weights")
axs[2].legend(["Positive", "Negative"])
axs[2].plot([sample_numbers[1], sample_numbers[-1]], [0, 0], '--k')
axs[2].set_ylabel("Total weight change")
axs[2].set_xlabel("Sample number")
plt.tight_layout()

# Plot the hit and miss rates over each epoch.
hit_miss_rate = np.load(os.path.join(foldername, filename_hit_miss_rate))
plt.figure()
plt.plot(np.arange(len(hit_miss_rate[:,0])), hit_miss_rate[:,0])
plt.plot(np.arange(len(hit_miss_rate[:,1])), hit_miss_rate[:,1])
plt.title("Hit and miss rates")
plt.xlabel("Epoch")
plt.ylabel("Rate")
plt.legend(["Hit rate", "Miss rate"])
plt.tight_layout()

plt.show()