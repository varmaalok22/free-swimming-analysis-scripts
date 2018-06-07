def bout_detect(raw_motion_data):
    """Input the raw motion data as a vector of numbers that correspond to the
    raw motion value for each frame. This function identifies indices where nonzero
    values lie. Using this as a starting point, it looks window_length indices ahead
    to find the first zero value. All the intermediate indices are considered a bout.
    The bouts are then filtered such that anything less than 5 frames is not considered
    a bout."""
    import numpy as np
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


def csvWrite(csvfile_path, fish_id, data):
    import csv
    import itertools
    import os
    csvName = csvfile_path.split('\\')[-1].split('.')[0]
    header = ['Fish ID', csvName]

    # The 'row' variable just has the Fish_ID and the corresponding data flattened.
    if not isinstance(data, list):
        data = [data]  # Just to take care of exceptional cases, such as bout_number.

    row = [[fish_id], data]
    row = list(itertools.chain(*row))

    if not os.path.isfile(csvfile_path):
        with open(csvfile_path, 'wb') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(header)

    with open(csvfile_path, 'ab') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
