#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 15:18:22 2021

@author: abhi
"""
import cv2
import numpy as np
from scipy import stats

img=cv2.imread("ref_marker.png")
print(img.shape)
img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)



img=cv2.resize(img,(80,80))
m,n=img.shape
#ret,img=cv2.threshold(img,150,255,cv2.THRESH_BINARY)
image=img.copy()
rev = []
resized_img = cv2.resize(img, (m,n), fx = 0, fy = 0, interpolation = cv2.INTER_CUBIC)
coordinate = [[3,3],[6,3],[6,6],[3,6]]
x_8 = int(m/8)
y_8 = int(n/8)
for i in coordinate:
    x = i[0]
    y = i[1]
    #print("i is ",i)
    pt_0 = int(img[(y_8 * y) - 0][(x_8 * x) - 0])
    pt_1 = int(img[(y_8 * y) - 1][(x_8 * x) - 1])
    pt_2 = int(img[(y_8 * y) - 2][(x_8 * x) - 2])
    pt_3 = int(img[(y_8 * y) - 3][(x_8 * x) - 3])
    pt_4 = int(img[(y_8 * y) - 4][(x_8 * x) - 4])
    pts=np.array([pt_0,pt_1,pt_2,pt_3,pt_4])
    #print(type(g))
    average= np.sum(pts)/len(pts)
    #print(average)

    if average > 200:

        break

index = coordinate.index(i)



if index == 0:
    print("AR tag is faces down")
    x = 5
    y = 5
    for i in range(4):
        pt_0 = int(img[(y_8 * y) - 0][(x_8 * x) - 0])
        pt_1 = int(img[(y_8 * y) - 1][(x_8 * x) - 1])
        pt_2 = int(img[(y_8 * y) - 2][(x_8 * x) - 2])
        pt_3 = int(img[(y_8 * y) - 3][(x_8 * x) - 3])
        pt_4 = int(img[(y_8 * y) - 4][(x_8 * x) - 4])
        pts=np.array([pt_0,pt_1,pt_2,pt_3,pt_4])
        #print(type(g))
        average= np.sum(pts)/len(pts)

        if i == 0:
            x = x - 1
        elif i == 1:
            y = y - 1
        elif i == 2:
            x += 1

        if average > 205:
            rev.append(1)
        else:
            rev.append(0)
    print("bits of the ar tag ",rev)
elif index == 1:
    print("AR tag  faces left")

    x = 4
    y = 5
    for i in range(4):
        pt_0 = int(img[(y_8 * y) - 0][(x_8 * x) - 0])
        pt_1 = int(img[(y_8 * y) - 1][(x_8 * x) - 1])
        pt_2 = int(img[(y_8 * y) - 2][(x_8 * x) - 2])
        pt_3 = int(img[(y_8 * y) - 3][(x_8 * x) - 3])
        pt_4 = int(img[(y_8 * y) - 4][(x_8 * x) - 4])
        pts=np.array([pt_0,pt_1,pt_2,pt_3,pt_4])
        #print(type(pts))
        average= np.sum(pts)/len(pts)

        if i == 0:
            y = y - 1
        elif i == 1:
            x += 1
        elif i == 2:
            y += 1

        if average > 205:
            rev.append(1)
        else:
            rev.append(0)
    print("bits of the ar tag ",rev)
elif index == 2:
    print("AR tag faces up")
    x = 4
    y = 4
    for i in range(4):
        pt_0 = int(img[(y_8 * y) - 0][(x_8 * x) - 0])
        pt_1 = int(img[(y_8 * y) - 1][(x_8 * x) - 1])
        pt_2 = int(img[(y_8 * y) - 2][(x_8 * x) - 2])
        pt_3 = int(img[(y_8 * y) - 3][(x_8 * x) - 3])
        pt_4 = int(img[(y_8 * y) - 4][(x_8 * x) - 4])
        pts=np.array([pt_0,pt_1,pt_2,pt_3,pt_4])
        #print(type(g))
        average= np.sum(pts)/len(pts)

        if i == 0:
            x += 1
        elif i == 1:
            y += 1
        elif i == 2:
            x = x - 1

        if average > 205:
            rev.append(1)
        else:
            rev.append(0)
    print("bits of the ar tag ",rev)
elif index == 3:
    print("AR Tag  faces right")
    x = 5
    y = 4
    for i in range(4):
        pt_0 = int(img[(y_8 * y) - 0][(x_8 * x) - 0])
        pt_1 = int(img[(y_8 * y) - 1][(x_8 * x) - 1])
        pt_2 = int(img[(y_8 * y) - 2][(x_8 * x) - 2])
        pt_3 = int(img[(y_8 * y) - 3][(x_8 * x) - 3])
        pt_4 = int(img[(y_8 * y) - 4][(x_8 * x) - 4])
        pts=np.array([pt_0,pt_1,pt_2,pt_3,pt_4])
        #print(type(g))
        average= np.sum(pts)/len(pts)

        if i == 0:
            y = y + 1
        elif i == 1:
            x = x - 1
        elif i == 2:
            y = y - 1

        if average > 205:
            rev.append(1)
        else:
            rev.append(0)
    print("bits of the ar tag ",rev)
