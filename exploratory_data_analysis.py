import mne
import os

from read_eeg import EEGData

if __name__ == "__main__":
    mne.set_log_level("WARNING")

    test = EEGData("chbmit", 1, "all")
    # seizure_data, non_seizure_data = test.get_and_split_data(prepend_seizure=0, save_non_seizure = True)

    save_dir = "/home/tim/SNN Seizure Detection/results"

    # fig = seizure_data.compute_psd(fmax=128).plot()
    # filename = "seizure_data.png"
    # save_path = os.path.join(save_dir, filename)
    # fig.savefig(save_path)

    # fig = non_seizure_data.compute_psd(fmax=128).plot()
    # filename = "non_seizure_data.png"
    # save_path = os.path.join(save_dir, filename)
    # fig.savefig(save_path)

    data = test.get_data()