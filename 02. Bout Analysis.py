# -*- coding: utf-8 -*-
"""
Analyze the Raw Motion Data to get Bout Information.

Created on Sun Feb 19 20:04:52 2017

@author: Aalok
"""

import numpy as np
import scipy.io as io  # Need it in order to load mat files
import matplotlib.pyplot as plt
import os
import cPickle as pickle
import timeit


def bout_detect(raw_motion_data):
    """Input the raw motion data as a vector of numbers that correspond to the
    raw motion value for each frame. This function identifies indices where nonzero
    values lie. Using this as a starting point, it looks window_length indices ahead
    to find the first zero value. All the intermediate indices are considered a bout.
    The bouts are then filtered such that anything less than 5 frames is not considered
    a bout."""
    window_length = 50  # Most bouts last much lesser than 20 frames,
    # so this is a really safe window to search for bouts.
    nonzero_indices = np.flatnonzero(raw_motion_data)
    all_bout_indices = []  # A list of all the indices that belong to all bouts.
    bout_indices = []  # A list of tuples of bout indices.
    bouts = []  # A list of bouts, stored as tuples.
    # Run through the list of nonzero indices and look for the first zero value
    # after it.
    do_not_append = False
    for i in nonzero_indices:
        if i not in all_bout_indices:
            bout_start = i  # Bout starting point
            try:
                first_zero_after_start = np.flatnonzero(
                    raw_motion_data[bout_start:bout_start+window_length+1] == 0)[0]
            except IndexError:
                try:
                    first_zero_after_start = np.flatnonzero(
                        raw_motion_data[bout_start:len(raw_motion_data)] == 0)[0]
                except IndexError:
                    first_zero_after_start = len(raw_motion_data)
                    do_not_append = True
            if not do_not_append:
                # Adjusted to start and end at 0
                bout = tuple(raw_motion_data[bout_start-1:bout_start+first_zero_after_start+1])
                bout_inds = tuple(range(bout_start-1, bout_start+first_zero_after_start+1))
                bout_indices.append(bout_inds)
                bouts.append(bout)
                for j in range(bout_start, bout_start+first_zero_after_start):
                    all_bout_indices.append(j)

    # Filter out the bouts list to remove tuples which are smaller than 5 entries long.
    """This is not really needed, because the fish can make some tiny movements
    that need to be detected and counted as bouts. In any case, the decision for this
    can be made much later. It is not really important to figure this out right now."""
    # bouts[:]        = [tup for tup in bouts if len(tup) > 5]
    # bout_indices[:] = [x for x in bout_indices if len(x) > 5]
    return bouts, bout_indices


def bout_params(bouts, bout_indices):
    """Takes a list of bout tuples as the input and estimates a series of parameters,
    such as mean number of bouts per unit time, mean bout duration,
    mean inter-bout interval. The function will also output all the values for
    each of these as a list, which can then be stored for later re-analysis purposes,
    if needed."""
    no_of_bouts = len(bouts)
    bout_durations = []
    interbout_intervals = []
    for ind, x in enumerate(bout_indices):
        bout_durations.append(len(x))  # Bout duration in frames
        if ind > 0:
            interbout_interval = x[0] - bout_indices[ind-1][-1]
            interbout_intervals.append(interbout_interval)
    return no_of_bouts, bout_durations, interbout_intervals


# directory = r'D:\VT Lab Stuff\Project 01 - Characterizing Auts2 Mutants\C. Free Swimming Experiments\Final Experiment\2017-03-23\10dpf'
directory = r'D:\VT Lab Stuff\Project 03 - Dopamine and auts2a\01. Free Swimming\WT\Fish_02\Motion Detection Videos'
filename = 'RawMotionInfo.mat'
path = os.path.join(directory, filename)
mat = io.loadmat(path)  # This loads the struct data as a dictionary.

"""
Some information on how to index into the mat dictionary. Firstly, the key for
the raw motion data is 'raw_motion_data', so all the data is stored in the
mat['raw_motion_data'] slice. This is an array with ALL the information, so you need
to subslice. You need three more indices to slice into the data. The first is rather
redundant, but it essentially converts the 'raw_motion_data' slice into a vector.
The vector is a vector of tuples. Each tuple stores the different fields of the structure.
In this case, the tuples are (filename, raw_motion).

In summary, a typical indexing looks like this: bouts = mat['raw_motion_data'][0][1][1]
"""

# Set any raw motion value below this to 0. Determined from the raw motion plot for 1000 frames.
threshold = 30
raw_motion_plot = False
bout_detection_plot = True


duration = 5 * 60  # seconds
frame_rate = 150  # fps (average)
endIndex = frame_rate * duration

raw_motion_data = {}
for index in range(len(mat['raw_motion_data'][0])):
    name = mat['raw_motion_data'][0][index][0]  # Fish ID
    motion = mat['raw_motion_data'][0][index][1]  # Motion Data
    motion = motion[:endIndex]
    motion[motion < threshold] = 0  # Threshold the motion data
    raw_motion_data[index] = tuple([name, motion])
    label = ' '.join(name[0].split('_')[:2])
    if raw_motion_plot:
        plt.plot(motion, label=label)
if raw_motion_plot:
    plt.legend()
    plt.title("Raw Motion Data")

write_raw_motion = False
if write_raw_motion:
    with open(os.path.join(directory, "Thresholded_Raw_Motion.pickle"), 'wb') as f:
        pickle.dump(raw_motion_data, f, protocol=pickle.HIGHEST_PROTOCOL)

# Calculate bout parameters for all the fish.
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

    # Extract bout parameters from a list of bouts.
    bout_number, bout_durations, interbout_intervals = bout_params(bouts, bout_indices)
    all_bout_numbers[fish_id] = bout_number
    all_bout_durations[fish_id] = bout_durations
    all_interbout_intervals[fish_id] = interbout_intervals
    # print bout_number

plt.figure()
bout_nums = [i[1] for i in all_bout_numbers.iteritems()]
# plt.boxplot(np.array(bout_durs)/float(frame_rate))
mean_no = np.mean(bout_nums)
std_no = np.std(bout_nums)
width = 0.5
plt.bar(1, mean_no, yerr=std_no, width=width, zorder=-1)
rand_x = (1-width/2) + (width)*np.random.rand(len(bout_nums))
plt.scatter(rand_x[:len(bout_nums)], bout_nums, color='k', zorder=1)
plt.xlim((0.4, 1.6))
plt.ylim((mean_no - 2*std_no, mean_no + 2*std_no))
plt.ylabel("Bout Number")
plt.title("Distribution of No of Bouts in 5 minutes")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')  # labels along the bottom edge are off


plt.figure()
bout_durs = [1000*np.array(all_bout_durations[i])/float(frame_rate)
             for i in all_bout_durations.iterkeys()]
# plt.boxplot(np.array(bout_durs)/float(frame_rate))
plt.boxplot(bout_durs)
plt.xlabel("Fish ID")
plt.ylabel("Bout duration (ms)")
plt.title("Bout Duration Distribution Per Fish")

plt.figure()
ibis = [1000*np.array(all_interbout_intervals[i])/float(frame_rate)
        for i in all_interbout_intervals.iterkeys()]
plt.boxplot(ibis)
plt.xlabel("Fish ID")
plt.ylabel("Inter-bout Interval (ms)")
plt.title("Interbout Interval Distribution Per Fish")

plt.figure()
pooled_bout_durs = []
for j in all_bout_durations.iteritems():
    pooled_bout_durs.extend(j[1])
plt.boxplot(1000*np.array(pooled_bout_durs)/float(frame_rate))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')  # labels along the bottom edge are off
plt.ylabel("Bout Duration (ms)")
plt.title("Bout Duration Distribution (WT, pooled)")

plt.figure()
pooled_ibis = []
for j in all_interbout_intervals.iteritems():
    pooled_ibis.extend(j[1])  # j[0] is the key, j[1] is the element
plt.boxplot(1000*np.array(pooled_ibis)/float(frame_rate))
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off')  # labels along the bottom edge are off
plt.ylabel("Interbout Interval (ms)")
plt.title("Interbout Interval Distribution (WT, pooled)")


mean_bout_durs = []
std_bout_durs = []
for i in bout_durs:
    mean_bout_durs.append(i.mean())  # This is the mean bout duration in MILLISECONDS
    std_bout_durs.append(i.std())

plt.bar(np.arange(1, len(mean_bout_durs)+1), mean_bout_durs, yerr=std_bout_durs)
plt.title(r"Bout Duration Per Fish (Mean $\pm$ SD)")
plt.xlabel("Fish ID")
plt.ylabel("Bout Duration (ms)")

mean_ibis = []
std_ibis = []
for j in ibis:
    mean_ibis.append(j.mean())  # This is the mean interbout interval in MILLISECONDS!!
    std_ibis.append(j.std())

plt.bar(np.arange(1, len(mean_ibis)+1), mean_ibis, yerr=std_ibis)
plt.title(r"Interbout Interval Per Fish (Mean $\pm$ SD)")
plt.xlabel("Fish ID")
plt.ylabel("Interbout Interval (ms)")

# Export all the bouts of each fish. From there, all other parameters can be extracted simply.
# The easiest way of doing this is not using CSV or these kinds of formats, but the PICKLE format.
# It's pretty cool, actually.
write_to_file = False
if write_to_file:
    pickle_filename = "All Bouts.pickle"
    pickle_file_path = os.path.join(directory, pickle_filename)
    with open(pickle_file_path, 'wb') as f:
        pickle.dump(all_bouts, f, protocol=pickle.HIGHEST_PROTOCOL)
