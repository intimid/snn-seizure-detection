class DataEncoder():
    def __init__(self, data: list, num_thresholds: int, data_range=None, norm_range=[0,1]):
        self.data = data
        self.num_thresholds = num_thresholds
        self.data_range = data_range


    def normalise_data(self, data, data_range, norm_range):
        """Apply min-max normalisation on the data.
        """
        # Define the range of the data and the range of the normalised data.
        if data_range is None:
            data_range = [min(data), max(data)]
        x_min = data_range[0]
        x_max = data_range[1]
        a = norm_range[0]
        b = norm_range[1]

        # Normalise the data.
        normalised_data = [None] * len(data)
        for i in range(len(data)):
            normalised_data[i] = a + ((data[i] - x_min) * (b - a)) / (x_max - x_min)

        return normalised_data


    def threshold_encode(self, data=None, ignore_outliers=True, outlier_thresholds=[-800,800]):
        """Encode the data by onset-offset thresholding.
        """
        if data == None:
            data = self.data

        # Define the range of the data and the range of the normalised data.
        if self.data_range is None:
            self.data_range = [min(data), max(data)]
        x_min = self.data_range[0]
        x_max = self.data_range[1]
        thr_interval = (x_max - x_min) / (self.num_thresholds + 1)

        # Determine the threshold values.
        thr_vals = [x_min + (i * thr_interval) for i in range(1, self.num_thresholds+1)]

        onset_times = {i: [] for i in range(self.num_thresholds)}
        offset_times = {i: [] for i in range(self.num_thresholds)}
        current_thr_zone = 0
        # Record each time a contiguous data point crosses a threshold.
        for i in range(len(data)):
            # Don't threshold outliers.
            if ignore_outliers:
                if data[i] <= outlier_thresholds[0] or data[i] >= outlier_thresholds[1]:
                    continue
            # Determine the treshold zone of the new data point.
            # The first threshold zone (0) is between x_min and thr_vals[0] and 
            # increases until the last threshold zone (num_thresholds+) which 
            # is between thr_vals[-1] and x_max.
            if data[i] <= x_min:
                # The first threshold zone is up to and including x_min.
                new_thr_zone = 0
            elif data[i] >= x_max:
                # The last threshold zone is x_max and above.
                new_thr_zone = self.num_thresholds
            else:
                new_thr_zone = (data[i] - x_min) // thr_interval

            # If a threshold is crossed, record the time as an onset or offset 
            # and update the current threshold zone.
            if new_thr_zone > current_thr_zone:
                # NOTE: The first data point is considered a threshold onset 
                #       for the immediately lower threshold zone.
                #       However,if the first data point occurs in the first 
                #       threshold zone, it is not considered a threshold onset 
                #       (nor offset).
                onset_times[int(new_thr_zone)-1].append(i)
                current_thr_zone = new_thr_zone
            if new_thr_zone < current_thr_zone:
                offset_times[int(new_thr_zone)].append(i)
                current_thr_zone = new_thr_zone

        return onset_times, offset_times


    def threshold_encode_multichannel(self, channels=None, ignore_outliers=True, outlier_thresholds=[-800,800]):
        """Threshold encode multichannel data.
        """
        if channels == None:
            channels = len(self.data)

        onset_times = [None] * channels
        offset_times = [None] * channels
        for i in range(channels):
            onset_times[i], offset_times[i] = self.threshold_encode(self.data[i], ignore_outliers, outlier_thresholds)

        return onset_times, offset_times


    def threshold_neuron_encode_multichannel(self, channels=None, fs=1, 
                                             ignore_outliers=True, outlier_thresholds=[-800,800]):
        """Threshold encode multichannel data with neuron information.
        """
        if channels == None:
            channels = len(self.data)
        onset_times, offset_times = self.threshold_encode_multichannel(channels, ignore_outliers, outlier_thresholds)

        T = 1 / fs  # Sampling period.
        input_indices = []
        input_times = []
        # For each channel, for each threshold, for each onset/offset time,
        # record the neuron index and the time.
        # Neuron indices are ordered first by channel, then by threshold, then
        # by onset/offset.
        for i in range(channels):
            for key, value in onset_times[i].items():
                if len(value) > 0:
                    idx = int(key + (self.num_thresholds*i*2) + key)
                    for j in range(len(value)):
                        input_indices.append(idx)
                        input_times.append(value[j]*T)
            for key, value in offset_times[i].items():
                if len(value) > 0:
                    idx = int(key + (self.num_thresholds*i*2) + key + 1)
                    for j in range(len(value)):
                        input_indices.append(idx)
                        input_times.append(value[j]*T)

        return input_indices, input_times



if __name__ == '__main__':
    # Test the DataEncoder class with random data.
    import random
    data = [random.randint(1, 100) for _ in range(10)]
    print(data)
    num_thresholds = 10
    encoder = DataEncoder(data, num_thresholds)
    # normalised_data = encoder.normalise_data(data, data_range=[0,100], norm_range=[0,1])
    # print(normalised_data)
    encoder.threshold_encode(data, data_range=[0,100], num_thresholds=9)