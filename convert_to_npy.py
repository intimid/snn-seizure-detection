import os
import re
import numpy as np

# # Set the path to the folder containing the numpy files.
# folder_path = "/home/tim/SNN Seizure Detection/TUH/tuh_data_preprocess/tuh_stft_ica_devpei12s"

# # Create empty lists to store the numpy arrays and labels.
# npy_arrays = []
# labels = []

# # Create regex pattern to identify seizure and non-seizure files.
# seizure_regex = re.compile(r"_seiz_")
# non_seizure_regex = re.compile(r"_bckg_")

# ctr = 1
# # Loop through each file in the folder.
# for file_name in os.listdir(folder_path):
#     if ctr % 100 == 0:
#         print(f"Processing file {ctr}...")
#     if file_name.endswith(".npy"):
#         # Load the numpy array from the file.
#         npy_array = np.load(os.path.join(folder_path, file_name))

#         if seizure_regex.search(file_name):
#             # If the file is a seizure file, set the label to 1.
#             label = 1
#         elif non_seizure_regex.search(file_name):
#             # If the file is a non-seizure file, set the label to 0.
#             label = 0
#         else:
#             # If the file is neither a seizure nor a non-seizure file, raise an error.
#             raise ValueError(f"File {file_name} is neither a seizure nor a non-seizure file.")

#         # Append the numpy array and label to their respective lists.
#         npy_arrays.append(npy_array)
#         labels.append(label)

#     ctr += 1

# # Convert the data and labels into numpy arrays.
# npy_arrays = np.array(npy_arrays)
# labels = np.array(labels)

# # Save the files.
# np.save("/home/tim/SNN Seizure Detection/TUH/tuh_stft_ica_devpei12s_npy/trainx.npy", npy_arrays)
# np.save("/home/tim/SNN Seizure Detection/TUH/tuh_stft_ica_devpei12s_npy/trainy.npy", labels)

# Check the outputs are correct.
data = np.load("/home/tim/SNN Seizure Detection/TUH/tuh_stft_ica_devpei12s_npy/trainx.npy")
labels = np.load("/home/tim/SNN Seizure Detection/TUH/tuh_stft_ica_devpei12s_npy/trainy.npy")

print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Print the number seizure and non-seizure files.
print(f"Number of seizure files: {np.sum(labels == 1)}")
print(f"Number of non-seizure files: {np.sum(labels == 0)}")

# Print the mean, median, max and min number of the data.
print(f"Mean of data: {np.mean(data)}")
# Get the percentiles.
print(f"Percentiles: {np.percentile(data, [0, 25, 50, 75, 100])}")