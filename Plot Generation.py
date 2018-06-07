import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# Plot data from the free swimming bout analysis.

# Step 3 - Plot the data
plt.figure()
bout_nums = [i[1] for i in all_bout_numbers.iteritems()]
# plt.boxplot(np.array(bout_durs)/float(frame_rate))
plt.boxplot(bout_nums)
plt.scatter(np.ones_like(bout_nums), bout_nums)
plt.show()
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
