#!/usr/bin/env python
# coding: utf-8

# In[1]:


from device import Camera
import cv2
from processors_noopenmdao import findFaceGetPulse
from interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
import socket
import sys
from matplotlib import pyplot as plt


# In[5]:


class getPulseApp():
    flagTest = True
    var = 0
    mean = 0


    def __init__(self):
        self.processor = findFaceGetPulse(bpm_limits=[50, 180],
                                          data_spike_limit=2500.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

    def toggle_search(self):
        """
        Toggles a motion lock on the processor's face detection component.

        Locking the forehead location in place significantly improves
        data quality, once a forehead has been sucessfully isolated.
        """
        state = self.processor.find_faces_toggle()
        print("face detection lock =", not state)

    def main_loop(self, i):
    
        self.bpm_plot = True
        """
        Single iteration of the application's main loop.
        """
        ret, frame = cap.read()
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _,thresh = cv2.threshold(img,10,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        offset_y = 0
        offset_x = 0
        frame = frame[y+offset_y:y+h-offset_y,x+offset_x:x+w-offset_x]
        self.h, self.w, _c = frame.shape

        # set current image frame to the processor's input
        self.processor.frame_in = frame
        self.processor.coords = frame
#         # process the image frame to perform all needed analysis
        self.processor.run('')
#         # collect the output frame for display
        output_frame = self.processor.frame_out
        
        if self.flagTest is True and i > 10:
            self.toggle_search()
            self.flagTest = False
        
        if (i % 80 == 0 or i % 81 == 0) and i > 10:
            self.toggle_search()
            self.flagTest = False
        self.var = np.var(self.processor.bpms)
        self.mean = np.mean(self.processor.bpms)


# In[6]:


App = getPulseApp()

cap = cv2.VideoCapture('./project.avi')
property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
video_length = int(cv2.VideoCapture.get(cap, property_id))



for i in range(video_length):
    App.main_loop(i)
print('var = ', App.var)
print('mean = ', App.mean)


# In[ ]:




