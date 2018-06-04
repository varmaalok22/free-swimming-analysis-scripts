# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 23:54:23 2017

Frame Subtraction - Free Swimming Experiments

@author: Aalok
"""

import cv2
import numpy as np
#import matplotlib.pyplot as plt
#import csv
import os
#import pandas as pd
import string
import timeit as ti

in_directory = r'D:\VT Lab Stuff\Project 01 - Characterizing Auts2 Mutants\C. Free Swimming Experiments\Final Experiment\2017-02-28\6dpf'
out_directory = os.path.join(in_directory, 'Processed Videos')
#out_directory = ''

if not os.path.exists(out_directory):
    os.mkdir(out_directory)

#input_filename = "Fish_01.mp4"
input_filename = "Fish_06.avi"
#output_filename = string.join(i for i in input_filename.split('_')[:2]) + '.avi'
output_filename = input_filename.split('.')[0] + ' (Frame Subtracted).avi'

input_path = os.path.join(in_directory, input_filename)
output_path = os.path.join(out_directory, output_filename)

video = cv2.VideoCapture(input_path)

ret, frame = video.read()
assert ret == True

video_width = int(video.get(3))
video_height = int(video.get(4))
fps = video.get(5)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height), False)


count = 0 # This is a variable to print, to keep track of progress
start_time = ti.default_timer()
while(ret):
    frame_1 = frame # Get the old frame, to be subtracted from the new one.
    
    ret, frame = video.read() # Read in the new frame.
    
    if not ret:
        break
    
    subtracted_frame = cv2.subtract(frame_1, frame)
    out.write(subtracted_frame)
    
    count += 1
    
    if count % 100 == 0:
        print str(count) + " Frames Processed."
        
    cv2.imshow('Frame Subtraction', subtracted_frame)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()

end_time = ti.default_timer()
elapsed_time = end_time - start_time
print "Time taken to process %s: %f seconds" %(output_filename[:-4], elapsed_time)