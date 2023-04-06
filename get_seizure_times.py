import pandas as pd

import re
from datetime import datetime as dt
from datetime import timedelta

summary_filename = "chb06-summary.txt"

seizure_seconds_regex = re.compile(r"(\d+) seconds")

def get_line_value(line: str):
    return line.split(":", 1)[1].strip()

def format_date(date: str):
    """Convert a date string to a datetime object.
    
    A date string in the format of HH:MM:SS is converted to a datetime object. 
    Incompatibilities and inconsistencies are also handled.
    """
    # Some hour values are incorrectly only one character long. So, instead of 
    # indexing the string, the split method is used for consistency.
    date_split = date.split(":", 1)
    hour = int(date_split[0])
    # If the hour value is greater than 23, it is assumed that the date has
    # wrapped around to the next day and so 24 is subtracted from the hour 
    # value.
    if hour > 23:
        date = str(hour-24) + ":" + date_split[1] 
    date = dt.strptime(date, '%H:%M:%S')

    return date

# Read the summary file for the EEG subject.
with open(summary_filename, 'r') as f:
    running_time = 0
    file_end_time = None
    seizure_times = {}

    while line := f.readline():
        # Get information about the current EEG file.
        # The order of the lines in the summary file is consistent and so the 
        # information may be read sequentially.
        if line.startswith("File Name:"):
            file_name = get_line_value(line)

            # Calculate the running time for the EEG files in seconds.
            file_start_time = format_date(get_line_value(f.readline()))
            file_end_time = format_date(get_line_value(f.readline()))
            file_duration = file_end_time - file_start_time
            # When the file end time wraps around to a new day, add a day to 
            # the duration.
            if file_duration < timedelta(seconds=0):
                file_duration += timedelta(days=1)

            # For files containing seizures, calculate the seizure start and 
            # end times.
            if num_seizures := int(get_line_value(f.readline())):  # num_seizures is a truthy value.
                # The next lines in the file describe the start and end times 
                # for the seizure(s).
                # TODO: This is untested for files with >1 seizures.
                for i in range(num_seizures):
                    # For the seizure start timestamp:
                    seizure_start_seconds = re.search(seizure_seconds_regex, get_line_value(f.readline())).group(1)
                    seizure_start_time = running_time + int(seizure_start_seconds)
                    # For the seizure end timestamp:
                    seizure_end_seconds = re.search(seizure_seconds_regex, get_line_value(f.readline())).group(1)
                    seizure_end_time = running_time + int(seizure_end_seconds)
                    # Add the seizure times to the dictionary.
                    seizure_times[file_name + f" {i+1}"] = (seizure_start_time, seizure_end_time)

            # Add the duration to the running time.
            running_time += file_duration.total_seconds()