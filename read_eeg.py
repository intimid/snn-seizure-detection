import numpy as np
import pandas as pd
import mne

import os
import json
import collections

from get_seizure_times import create_seizure_file

class EEGData:
    dataset_configs = None  # Class variable to store dataset configurations.

    # __slots__ is used instead of __dict__ to reduce memory usage.
    __slots__ = ('dataset', 'subject', '_channels', 'exclude', 'seizure_times')

    def __init__(self, dataset, subject, channels=None, exclude=None):
        self.dataset = dataset
        self.subject = subject
        self.channels = channels
        self.exclude = exclude

        # Initialise the dataset configurations if they have not been loaded.
        if not EEGData.dataset_configs:
            with open("/home/tim/SNN Seizure Detection/utils/dataset_configs.json", 'r') as f:
                EEGData.dataset_configs = json.load(f)

    @property
    def channels(self):
        return self._channels

    @channels.setter
    def channels(self, channels):
        """Sets the channels to be used for the EEG data based on user input.
        """
        # TODO: Consider using a .json file to store the channel names.
        if channels is None:  # Use all available EEG channels.
            pass
        elif channels == "SE":  # Using the channel with the lowest Sample Entropy (SE) values.
            pass
        else:
            self._channels = channels

    def __str__(self):
        return f"EEGData(dataset={self.dataset}, subject={self.subject}, channels={self.channels})"

    def chb_get_subject_folderpath(self, subject):
        dataset_dir = EEGData.dataset_configs["chbmit"]["directory"]
        subject_folder = f"chb{subject:02d}"

        return os.path.join(dataset_dir, subject_folder)

    def chb_get_seizure_times(self, subject):
        """Get the seizure times for the specified subject from a precomputed 
        .csv file. If the file does not exist, compute and store the seizure 
        times using the summary file."""
        # Check if the seizure times file exists.
        folderpath = "/home/tim/SNN Seizure Detection/utils/CHB-MIT Seizure Times"
        subject_folder = f"chb{subject:02d}"
        filepath = os.path.join(folderpath, f"{subject_folder}_seizure_times.csv")
        if not os.path.exists(filepath):
            create_seizure_file(subject, filepath)

        self.seizure_times = pd.read_csv(filepath)

    def get_edf_filepaths(self):
        """Get the filepaths of all .edf files for the specified subject."""
        folderpath = self.chb_get_subject_folderpath(self.subject)
        filepaths = [filename.path for filename in os.scandir(folderpath) if filename.name.endswith(".edf")]

        return filepaths

    def get_data(self):
        self.chb_get_seizure_times(self.subject)
        filepaths = self.get_edf_filepaths()

        # TODO: Use the "include" argument if channels is specified.
        data = [mne.io.read_raw_edf(filepath, preload=True).notch_filter(freqs=(60, 120)) for filepath in filepaths]
        data = mne.concatenate_raws(data)

        return data

    def get_and_split_data(self, prepend_seizure=0, save_non_seizure = True):
        """Get the EEG data for the specified subject and split it into seizure
        and non-seizure data."""
        self.chb_get_seizure_times(self.subject)
        filepaths = self.get_edf_filepaths()

        # Move seizure_data initialisation to after seizure_times has been made 
        # so that you can initialise it with the correct size.
        seizure_data = []
        if save_non_seizure:
            non_seizure_data = []
        for filepath in filepaths:
            raw = mne.io.read_raw_edf(filepath, preload=True).notch_filter(freqs=(60, 120))

            # Get the file name.
            filename = os.path.basename(filepath)
            # Check if the file is a seizure file.
            if filename in self.seizure_times['File'].values:
                seizure_times_df = self.seizure_times[self.seizure_times['File'] == filename]
                # Get the start and end times of all seizures in the file.
                seizure_times = [list(i) for i in zip(seizure_times_df['File Start Time (s)'].values, 
                                                        seizure_times_df['File End Time (s)'].values)]

                # Change the seizure_times to include the prepended seizure 
                # time if it is greater than 0.
                if prepend_seizure:
                    prev_seizure_end = None
                    for i in range(len(seizure_times)):
                        start, end = seizure_times[i]
                        # If the prepended seizure time is greater than what is available, reduce the prepended 
                        # seizure time to the maximum available time. The available time is defined as any 
                        # non-seizure data.
                        # If the first seizure occurs earlier than the prepended seizure time:
                        if start < prepend_seizure:
                            seizure_times[i][0] = 0
                            print(f"Prepended data for Seizure {i+1} in {filename} has been reduced to \
                                    {start} seconds.")
                        # If the gap between the current seizure and the previous seizure is greater than the 
                        # prepended seizure time:
                        elif prev_seizure_end is not None:
                            if start - prev_seizure_end < prepend_seizure:
                                seizure_times[i][0] = prev_seizure_end
                                print(f"Prepended data for Seizure {i+1} in {filename} has been reduced to \
                                        {start - prev_seizure_end} seconds.")
                        # There is sufficient non-seizure data to prepend the seizure data:
                        else:
                            seizure_times[i][0] = start - prepend_seizure

                if save_non_seizure:
                    # Get the start and end times of all non-seizures periods in the file.
                    non_seizure_times = list(zip(np.insert(seizure_times_df['File End Time (s)'].values, 0, 0), 
                                                    np.append(seizure_times_df['File Start Time (s)'].values, 
                                                            raw.tmax)))
                    # Remove all non-seizure periods that are 0 seconds long.
                    # I.e. The start and end times are the same.
                    non_seizure_times = [times for times in non_seizure_times if times[0] != times[1]]

                # Aggregate all seizure data for the file into the 
                # aggregate seizure_data list.
                for start, end in seizure_times:
                    seizure_data.append(raw.copy().crop(start, end))

                if save_non_seizure:
                    # Append all non-seizure data for the file into the 
                    # aggregate non_seizure_data list.
                    for start, end in non_seizure_times:
                        non_seizure_data.append(raw.copy().crop(start, end))

        seizure_data = mne.concatenate_raws(seizure_data)
        non_seizure_data = mne.concatenate_raws(non_seizure_data)

        return seizure_data, non_seizure_data




        # print(continuous_data.shape)



# foldername = "D:/Uni/Yessir, its a Thesis/Data/chb01"

# filepaths = [filename.path for filename in os.scandir(foldername) if filename.name.endswith(".edf")]
# continuous_data = [mne.io.read_raw_edf(filepath, preload=True).notch_filter(freqs=(60, 120)) for filepath in filepaths]

# continuous_data = mne.concatenate_raws(continuous_data)
# continuous_data.compute_psd(fmax=128).plot()
# continuous_data.plot(duration=5, n_channels=30, block=True)