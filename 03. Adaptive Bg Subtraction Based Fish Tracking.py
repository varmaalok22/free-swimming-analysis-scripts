# -*- coding: utf-8 -*-
"""
Free Swimming Tracking

Created on Mon Feb 20 11:56:16 2017

@author: Aalok
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

in_directory = r"D:\VT Lab Stuff\Project 03 - Dopamine and auts2a\01. Free Swimming\WT"

out_directory = os.path.join(in_directory, 'Background Subtraction Videos')
if not os.path.isdir(out_directory):
    os.makedirs(out_directory)

#All the mp4 or avi files in the directory:
#mp4_files = glob.glob(os.path.join(in_directory, '*.mp4'))
avi_files = glob.glob(os.path.join(in_directory, '*.avi'))

#for x in mp4_files:
for x in avi_files:
    input_filename = x.split('\\')[-1]
    output_filename = ' '.join(input_filename.split('_')[:2]) + ' (Bg subt).avi'
    output_filename = input_filename.split('.')[0] + ' (Bg subt).avi'
    print output_filename

    input_path = os.path.join(in_directory, input_filename)
    output_path = os.path.join(out_directory, output_filename)

    video = cv2.VideoCapture(input_path)
    ret, frame = video.read()
    assert ret == True
    
    video_width = int(video.get(3))
    video_height = int(video.get(4))
    fps = video.get(5) # The get feature has many properties.
    # The one with index 5 is the frame rate. It can be inaccurate at times
    # but I have verified that it works before I used it.
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (video_width, video_height), False)
    fgbg = cv2.createBackgroundSubtractorKNN()
    
    count = 0 # This is a variable to print, to keep track of progress
    while(ret):
        ret, frame = video.read()
        if not ret:
            break
        
        fgmask = fgbg.apply(frame)
        out.write(fgmask)
        count += 1
        
        if count % 5000 == 0:
            print str(count) + " Frames Processed."
            
        cv2.imshow('Bg subtracted frame', fgmask)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q'):
            break
        
    video.release()
    out.release()
    cv2.destroyAllWindows()