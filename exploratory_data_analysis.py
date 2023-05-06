import numpy as np
import matplotlib.pyplot as plt
# import mne

import os

from read_eeg import EEGData


def get_TUH_npy_data(filename, mmap_mode=None):
    """Gets the TUH .npy data that has been processed by Luis.

    mmap_mode is passed to np.load() to allow for memory mapping for 
    particularly large files.
    """
    data_dir = "/home/tim/SNN Seizure Detection/TUH/reshuffle"

    return np.load(os.path.join(data_dir, filename), mmap_mode=mmap_mode)

def get_sz_idxs(labels):
    """Gets the indices of the seizure and non-seizure data."""
    return np.where(labels == 1)[0], np.where(labels == 0)[0]

def split_TUH_sz_data(data, labels):
    """Splits the data into seizure and non-seizure data."""
    # Get the indices of the seizure data.
    sz_indices = np.where(labels == 1)[0]
    # Get the indices of the non-seizure data.
    non_sz_indices = np.where(labels == 0)[0]

    # Split the data.
    sz_data = data[sz_indices]
    non_sz_data = data[non_sz_indices]

    return sz_data, non_sz_data

def combine_eeg_data(data):
    """Combines the EEG data from one electrode."""
    # Get the number of samples and the size of each sample.
    num_samples = data.shape[0]
    sample_size = data.shape[1]

    # Initialise the combined data array.
    combined_data = np.empty(num_samples * sample_size)

    # Combine the data.
    for i in range(num_samples):
        combined_data[i*sample_size : (i + 1)*sample_size] = data[i]

    return combined_data

def get_fft(signal, fs, n):
    """Calculate the FFT of the EEG data."""
    # Calculate the real FFT.
    sp = np.fft.rfft(signal, n=n)
    # Get the frequency bins for the real FFT.
    freq = np.fft.rfftfreq(n=n, d=1/fs)

    return sp, freq

def plot_comparison_fft(sz_fft_dict, non_sz_fft_dict, electrode_group, electrode_list, n, save_dir):
    """Plots the seizure and non-seizure FFTs of the electrode group."""
    plt.subplots_adjust(hspace=0.8)
    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True)

    left_electrodes = []
    right_electrodes = []
    for electrode in electrode_list:
        # Get the seizure and non-seizure FFT data.
        sz_sp, sz_freq = sz_fft_dict[electrode]
        sz_sp = abs(sz_sp) / n  # Get the magnitude of the amplitude and normalise it.
        non_sz_sp, non_sz_freq = non_sz_fft_dict[electrode]
        non_sz_sp = abs(non_sz_sp) / n  # Get the magnitude of the amplitude and normalise it.

        # Plot the FFTs onto the first, aggregate figure.
        axs[0,0].plot(non_sz_freq, non_sz_sp, linewidth=0.5)
        axs[0,1].plot(sz_freq, sz_sp, linewidth=0.5)
        # If the electrode is in the midline, plot it on all figures.
        if electrode[-1] == 'z':  # z means midline.
            axs[1,0].plot(non_sz_freq, non_sz_sp, linewidth=0.5)
            axs[2,0].plot(non_sz_freq, non_sz_sp, linewidth=0.5)
            axs[1,1].plot(sz_freq, sz_sp, linewidth=0.5)
            axs[2,1].plot(sz_freq, sz_sp, linewidth=0.5)
            left_electrodes.append(electrode)
            right_electrodes.append(electrode)
        # If the electrode is in the left hemisphere, plot it on the second figure.
        elif int(electrode[-1]) % 2 == 1:  # Odd number means left hemisphere.
            axs[1,0].plot(non_sz_freq, non_sz_sp, linewidth=0.5)
            axs[1,1].plot(sz_freq, sz_sp, linewidth=0.5)
            left_electrodes.append(electrode)
        # If the electrode is in the right hemisphere, plot it on the third figure.
        else:  # Even number means right hemisphere.
            axs[2,0].plot(non_sz_freq, non_sz_sp, linewidth=0.5)
            axs[2,1].plot(sz_freq, sz_sp, linewidth=0.5)
            right_electrodes.append(electrode)

    # Shade key brain waves.
    brain_waves = {"Delta": (0.5, 4), "Theta": (4, 7), "Alpha": (8, 12), "Beta": (13, 30)}
    shade_colours = {"Delta": "blue", "Theta": "green", "Alpha": "orange", "Beta": "red"}
    for ax in axs.flat:
        for brain_wave, (lower, upper) in brain_waves.items():
            ax.axvspan(lower, upper, color=shade_colours[brain_wave], alpha=0.05)

    # Set the limits of the x and y axes.
    for ax in axs.flat:
        ax.set_xlim(0, sz_freq[-1])
        ax.set_ylim(0, 8)
        ax.minorticks_on()
        ax.grid(True)

    # Add the legend.
    axs[0,0].legend(electrode_list)
    axs[0,1].legend(electrode_list)
    axs[1,0].legend(left_electrodes)
    axs[1,1].legend(left_electrodes)
    axs[2,0].legend(right_electrodes)
    axs[2,1].legend(right_electrodes)

    # Add the title and axis labels.
    fig.suptitle(f"FFT of {electrode_group} Electrodes", fontsize=20)
    axs[0,0].set_title("All Electrodes (Non-Seizure)")
    axs[0,1].set_title("All Electrodes (Seizure)")
    axs[1,0].set_title("Left Hemisphere Electrodes (Non-Seizure)")
    axs[1,1].set_title("Left Hemisphere Electrodes (Seizure)")
    axs[2,0].set_title("Right Hemisphere Electrodes (Non-Seizure)")
    axs[2,1].set_title("Right Hemisphere Electrodes (Seizure)")
    axs[2,0].set_xlabel("Frequency (Hz)")
    axs[2,1].set_xlabel("Frequency (Hz)")
    axs[0,0].set_ylabel("|Amplitude|")
    axs[1,0].set_ylabel("|Amplitude|")
    axs[2,0].set_ylabel("|Amplitude|")

    # Add brain wave annotations.
    for ax in axs.flat:
        for brain_wave, (lower, upper) in brain_waves.items():
            ax.annotate(f"{brain_wave} Wave", xy=(upper - 0.75, 8 - 0.125), xycoords="data", 
                        color = shade_colours[brain_wave], 
                        rotation=-90, horizontalalignment="right", verticalalignment="top")

    # Save the figure.
    fig.set_size_inches(24, 15)
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(os.path.join(save_dir, f"FFT of {electrode_group} Electrodes.png"))
    plt.close(fig)



if __name__ == "__main__":
    # mne.set_log_level("WARNING")

    results_dir = "/home/tim/SNN Seizure Detection/results"

    # Get the raw TUH data.
    trainx = get_TUH_npy_data("trainx.npy", mmap_mode='r')
    trainy = get_TUH_npy_data("trainy.npy")
    # Get the indices of the seizure and non-seizure data.
    sz_idxs, non_sz_idxs = get_sz_idxs(trainy)

    # Get the number of seizure and non-seizure samples.
    num_sz = len(sz_idxs)
    num_non_sz = len(non_sz_idxs)

    electrode_labels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", 
                        "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", 
                        "O2"]
    sz_fft = dict.fromkeys(electrode_labels, None)
    non_sz_fft = dict.fromkeys(electrode_labels, None)

    fs = 250  # Sampling frequency.
    # Calculate the FFT of each electrode in the seizure data.
    for electrode in range(19):  # 19 electrodes are used.
        # Get the seizure EEG data and combine it.
        sz_eeg_data = trainx[sz_idxs, :, electrode]
        sz_combined_data = combine_eeg_data(sz_eeg_data)

        # Set the number of sample points for the FFT.
        n = 1000
        # Get the FFT of the seizure data and add it to the dictionary.
        sz_fft[electrode_labels[electrode]] = get_fft(sz_combined_data, fs, n)
        # Delete the EEG data to free up memory.
        del sz_eeg_data, sz_combined_data

        # Get the non-seizure EEG data and combine it.
        non_sz_eeg_data = trainx[non_sz_idxs, :, electrode]
        non_sz_combined_data = combine_eeg_data(non_sz_eeg_data)
        # Calculate the number of samples in the non-seizure data.
        N = non_sz_combined_data.shape[0]
        # Get the FFT of the non-seizure data and add it to the dictionary.
        non_sz_fft[electrode_labels[electrode]] = get_fft(non_sz_combined_data, fs, n)
        # Delete the EEG data to free up memory.
        del non_sz_eeg_data, non_sz_combined_data

    # Group the electrode by brain lobe and plot their respective FFTs.
    electrode_groups = {"Frontal": ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"], 
                        "Temporal": ["T3", "T4", "T5", "T6"], 
                        "Central": ["C3", "C4", "Cz"], 
                        "Parietal": ["P3", "P4", "Pz"], 
                        "Occipital": ["O1", "O2"]}
    frontal_electrodes   = ["Fp1", "Fp2", "F7", "F8", "F3", "F4", "Fz"]
    temporal_electrodes  = ["T3", "T4", "T5", "T6"]
    central_electrodes   = ["C3", "C4", "Cz"]
    parietal_electrodes  = ["P3", "P4", "Pz"]
    occipital_electrodes = ["O1", "O2"]

    # Plot the FFTs of each electrode group.
    for electrode_group, electrode_list in electrode_groups.items():
        plot_comparison_fft(sz_fft, non_sz_fft, electrode_group, electrode_list, n, results_dir)



    # Validate the EDA results.
    validx = get_TUH_npy_data("validx.npy")
    validy = get_TUH_npy_data("validy.npy")