# -*- coding: utf-8 -*-
"""
Fourier Transform Analysis of Bout Data

Created on Mon Mar 27 10:53:59 2017

@author: Aalok
"""

## Understanding the Fast Fourier Transform.
## It is a kind of DISCRETE Fourier Transform, which is used for DISCRETE signals.
## Since the bout data is just a single sequence of real numbers that represent the motion
## fluctuations of the fish, input is A ONE-DIMENSIONAL ARRAY OF REAL NUMBERS.
## Another very important parameter for the FFT is the sampling rate. The data was sampled
## at a rate of Fs, and that is one of the inputs to the FFT. Be sure about what the Fs is
## before applying the FFT.
## The OUTPUT of the Discrete Fourier Transform (which is calculated using the FFT algorithm)
## is a sequence of COMPLEX NUMBERS, i.e. a sequence of (Re, Im).
## If the sampling rate is Fs, the frequencies outputted by the FFT are in the range (0, Fs/2) INCLUSIVE!
## The Power Spectral Density is the following: 20*log10(sqrt(Re^2 + Im^2)), i.e. 20*log10(magnitude of cx no)

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
import os
import pandas as pd
import seaborn as sns

# Load all the bouts data
directory = r'D:\VT Lab Stuff\Project 01 - Characterizing Auts2 Mutants\C. Free Swimming Experiments\Final Experiment\2017-02-28\6dpf'
filename = "All Bouts.pickle"
path = os.path.join(directory, filename)

with open(path, 'rb') as f:
    all_bouts = pickle.load(f)
    
# For each fish and each bout, do the FFT, and store it in a dictionary
Fs = 150
single_bout = False
if single_bout: # Trial for one bout:
    bout = np.array(all_bouts[0][6])
    ps = np.fft.fft(bout)
    time_step = 1/Fs
    freqs = np.fft.fftfreq(bout.size, time_step)
    idx = np.argsort(freqs)
    
    plt.plot(freqs[idx], ps[idx])
    plt.title('Frequency Components of One Bout')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (verify units)')
    plt.xlim((0, Fs/2))

all_fft = {}
max_freq_per_bout = {}
for fish_id in all_bouts.iterkeys():
    all_fft[fish_id] = []
    max_freq_per_bout[fish_id] = []
    for index, bouts in enumerate(all_bouts[fish_id]):
        if len(bouts) > 5:
            bout = np.array(bouts)
            ps = np.fft.fft(bout)
            time_step = 1/Fs
            freqs = np.fft.fftfreq(bout.size, time_step)
            idx = np.argsort(freqs)
            # Sort out the frequencies and their corresponding powers, and analyse the POSITIVE frequencies only
            # to extract the maximum frequency component per bout.
            pos_freqs_loc = np.where(freqs >= 0)
            pos_freqs = freqs[pos_freqs_loc]
            pos_freqs_ps = ps[pos_freqs_loc]
            max_idx = np.argmax(pos_freqs_ps)
            max_freq_per_bout[fish_id].append(pos_freqs[max_idx]) # The index is not needed.
            all_fft[fish_id].append((index, freqs[idx], ps[idx]))
            #plt.plot(freqs[idx], ps[idx])

max_freq_per_bout_boxplot = True
if max_freq_per_bout_boxplot:
    plt.figure()
    max_freqs = pd.DataFrame.from_dict(max_freq_per_bout, orient='index')
    max_freqs = max_freqs.T
    max_freqs.columns = xrange(1, len(max_freqs.columns)+1)
    #max_freqs.T.boxplot()
    sns.boxplot(data=max_freqs)
    sns.stripplot(data=max_freqs, size=5, jitter=True, edgecolor='k', linewidth=0.5)
    plt.xlabel("Fish ID")
    plt.ylabel("Intra-Bout Frequency (Hz)")
    plt.title("Intra-Bout Frequency, Per Fish")

mean_tbf_plot = True
if mean_tbf_plot:
    avg_mean_max_freq = max_freqs.mean().mean()
    std_mean_max_freq = max_freqs.mean().std()
    plt.errorbar(0, avg_mean_max_freq, std_mean_max_freq, fmt='o', capsize=20, capthick=3)
    plt.ylim((0, Fs/2))
    plt.xticks([0], [''])
    plt.ylabel("Mean Bout Frequency (Hz)")
    plt.title('Mean Bout (Tail-Beat) Frequency')
    text_start_point_y = 5 + np.ceil(5 * round((avg_mean_max_freq + std_mean_max_freq)/5))
    plt.text(-.03, text_start_point_y, 'Mean $\pm$ SD = %s $\pm$ %s' %(str(round(avg_mean_max_freq, 2)), str(round(std_mean_max_freq, 2))))


with open(os.path.join(directory, "Thresholded_Raw_Motion.pickle"), 'rb') as f:
    raw_motion_data = pickle.load(f)

spectrogram = False
if spectrogram:
    raw_motion = raw_motion_data[0][1] # For Fish 01
    nfft = 15 #int(np.floor(np.log2(len(raw_motion))))
    fig, ax = plt.subplots(2, sharex=True)
    # The specgram function scales down the time by the sampling frequency, thereby reducing the total
    # number of entries. So it is best to pass Fs=1. That scales everything properly.
    ax[0].specgram(raw_motion.flatten(), NFFT=nfft, Fs=1, noverlap=nfft/2, window=np.hanning(nfft))
    ax[0].set_yticks(np.arange(0,0.5,0.1))
    ax[0].set_yticklabels((150*np.arange(0,0.5,0.1)).astype(np.int32))
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title("Spectrogram of Raw Motion Data from Fish 01")
    ax[1].plot(np.arange(0,len(raw_motion)), raw_motion)
    ax[1].set_xlabel("Time (mins)")
    ax[1].set_xticks((150*60)*np.arange(0, 5.1, 0.5))
    ax[1].set_xticklabels(np.arange(0, 5.1, 0.5))
    ax[1].set_ylabel("Raw Motion Value (a.u.)")
