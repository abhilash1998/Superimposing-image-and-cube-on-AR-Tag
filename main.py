#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 19:37:21 2021

@author: abhi
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
import scipy
import imutils

import Tag_detection_tracking

import os
#FPS_val = 25
out = cv2.VideoWriter('cube.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1920,1080))

file_location=os.getcwd()   #gets the current directory
video=cv2.VideoCapture(file_location+"/Tag1.mp4")
testudo=cv2.imread(file_location+"/testudo.png")
test_do1=testudo.copy()
cv2.imshow("testudo_",testudo)
tag=Tag_detection_tracking.Tag_detection_tracking()


cube_pts=np.array([[0,0,0,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1]])

while (video.isOpened()):
    ret,frame=video.read()
    if frame is None:
        break
    src_pts,extracted_ar_tag1,screenCnt,extraction_1,counter,frame4=tag.detecting_tag(frame)

    extracted_ar_tag=extracted_ar_tag1.copy()
    extracted_ar_tag2=extracted_ar_tag1.copy()

    dest_pts_t=np.array([[0,0],[0,500],[500,500],[500,0]])
    if counter==1:


        dest_pts=np.array([[0,0],[80,0],[80,80],[0,80]])
        H=tag.homography(src_pts,dest_pts)

        #cv2.imshow("output_of_detection",extracted_ar_tag)

        AR_tag3=tag.wrapping(extracted_ar_tag.copy(),H,screenCnt)

        #cv2.imshow("work",AR_tag1)




        orientation=tag.orientation(AR_tag3)
        if orientation == 0:
            dest_pts_t = np.array([[500,0], [0,0],[0,500], [500,500]])
        elif orientation == 1:
            dest_pts_t = np.array([[500,500],[500,0],[0,0],[0,500]])
        elif orientation ==2:
            dest_pts_t = np.array([[0,500],[500,500],[500,0],[0,0]])
        elif orientation == 3:
            dest_pts_t = np.array([[0,0],[0,500],[500,500],[500,0]])
        H_test=tag.homography(dest_pts_t,src_pts)

        #testudo_frame=tag.image_transformation(testudo,frame4.copy(),H_test)
        #cv2.imshow("testudo_frame",testudo_frame)
        #out.write(testudo_frame)
        #print(testudo_frame.shape)
        #img=AR_tag3
        H_cube=tag.homography(np.array([[0,0],[0,1],[1,1],[1,0]]),src_pts)
        #print(H_cube)
        cube_output=tag.cube(frame4.copy(),H_cube,cube_pts)

        out.write(cube_output)




        cv2.imshow("AR_tag_mycode",AR_tag3)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()
