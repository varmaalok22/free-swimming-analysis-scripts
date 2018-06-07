import numpy as np
import scipy.io as io  # Need it in order to load mat files
import matplotlib.pyplot as plt
import os
from analysisFunctions import bout_detect, bout_params, csvWrite

# directory = r'D:\VT Lab Stuff\Project 01 - Characterizing Auts2 Mutants\C. Free Swimming Experiments\Final Experiment\2017-03-23\10dpf'
directory = r'D:\VT Lab Stuff\Project 03 - Dopamine and auts2a\01. Free Swimming\WT\Fish_02\Motion Detection Videos'
filename = 'RawMotionFiltered.mat'
duration = 5 * 60  # Duration of the video in seconds
frame_rate = 150  # Frame rate (frames per second)

raw_motion_plot = False
bout_detection_plot = False

path = os.path.join(directory, filename)
mat = io.loadmat(path)  # This loads the struct data as a dictionary.

endIndex = frame_rate * duration

# Step 1 - Load all the data from the raw motion mat file (from MATLAB analysis)
raw_motion_data = {}
for index in range(len(mat['raw_motion_filtered'][0])):
    name = mat['raw_motion_filtered'][0][index][0]  # Fish ID
    motion = mat['raw_motion_filtered'][0][index][1]  # Motion Data
    motion = motion[:endIndex]
    raw_motion_data[index] = tuple([name, motion])
    label = ' '.join(name[0].split('_')[:2])
    if raw_motion_plot:
        plt.plot(motion, label=label)
if raw_motion_plot:
    plt.legend()
    plt.title("(Filtered) Raw Motion Data")

# Step 2 - Calculate bout parameters for all the fish, and save all this data.
all_bouts = {}
all_bout_numbers = {}
all_bout_indices = {}
all_bout_durations = {}
all_interbout_intervals = {}
for j in raw_motion_data.iterkeys():
    fish_id = j
    motion = raw_motion_data[fish_id][1]
    bouts, bout_indices = bout_detect(motion)
    all_bouts[fish_id] = bouts
    all_bout_indices[fish_id] = bout_indices

    # Plot the motion and highlight the detected bouts
    if bout_detection_plot:
        plt.figure()
        plt.plot(motion)
        for i in bout_indices:
            plt.axvspan(i[0], i[-1], facecolor='#2ca02c', alpha=0.5)
        plt.title("Bout detection for Fish %d" % (fish_id + 1))
        plt.xlabel("Time (frames)")
        plt.ylabel("Raw Motion (a.u.)")
        plt.show()

    # Extract bout parameters from a list of bouts.
    bout_number, bout_durations, interbout_intervals = bout_params(bouts, bout_indices)

    # The 'bouts' variable is not stored, because of its slightly complex structure.
    # However, the bout indices are.
    bout_indices_path = os.path.join(directory, 'Bout Indices.csv')
    bout_number_path = os.path.join(directory, 'Bout Number.csv')
    bout_durations_path = os.path.join(directory, 'Bout Durations.csv')
    interbout_intervals_path = os.path.join(directory, 'Interbout Intervals.csv')

    csv_paths = [bout_indices_path, bout_number_path, bout_durations_path, interbout_intervals_path]
    csv_data = [bout_indices, bout_number, bout_durations, interbout_intervals]
    for index, csvfile_path in enumerate(csv_paths):
        csvWrite(csvfile_path, fish_id, csv_data[index])

    all_bout_numbers[fish_id] = bout_number
    all_bout_durations[fish_id] = bout_durations
    all_interbout_intervals[fish_id] = interbout_intervals
    # print bout_number
