#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 14:53:02 2021

@author: abhi
"""
import numpy as np
import scipy
import cv2
import imutils
import math
class Tag_detection_tracking:
    def detecting_tag(self,frame):
        """
        Takes the fourier transformation of the image to remove the noise from the image
        and then inverse fourier transformation is taken so that which gives the output image
        which is then used to extract AR tag

        Parameters
        ----------
        frame : np.array /img
            Input image from the video

        Returns
        -------
        screenCnt : List
            corners off AR tag

        extraction_1 : np.array
            extracted AR tag
        rect : np.array
            Sortered array of corner used for homography
        frame3,frame4:np.array
            Copy of orignal frome
        screenCnt : np.array
            Contour points
        counter : Int
            To check if the AR tag is detected on not

        """
        counter=0
        #m=640
        #n=480
        m,n,color=frame.shape
        center=[m/2 , n/2]
        screenCnt=np.zeros((4,2)).astype(np.int32)
        frame4=frame.copy()
        #print(m,n)
        #frame = cv2.resize(frame,(640,480))
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame1=frame.copy()
        frame2=frame.copy()
        frame3=frame.copy()
        frame5=frame.copy()
        #ret,frame=cv2.threshold(frame,150,255,cv2.THRESH_BINARY)
        #cv2.imshow("frame",frame)
        frame = scipy.fft.fft2(frame)
        fshift = scipy.fft.fftshift(frame)


        #mask[masking]=0
        #plt.imshow(magnitude_spectrum,cmap="gray")
        crow=int(m/2)
        ccol=int(n/2)
        mask = np.ones((m,n),np.uint8)
        mask[crow-8:crow+8, ccol-8:ccol+8] = 0
        #mask[crow-2:crow+2, ccol-2:ccol+2] = 0
        #mask[crow-70:crow+70, ccol-70:ccol+70] = 0
        #print(fshift.shape,mask.shape)
        fshift = fshift*mask
        #fshift=cv2.bitwise_and(fshift,mask)
        #fshift[crow-20:crowq+20, ccol-20:ccol+20] = 0
        f_ishift = scipy.fft.ifftshift(fshift)

        img_back = scipy.fft.ifft2(f_ishift)
        img_back = (np.abs(img_back))
        img_back=img_back.astype(np.uint8)


        ret,img_back=cv2.threshold(img_back,60,255,cv2.THRESH_BINARY)
        frame1=cv2.bitwise_and(frame1,img_back)



    #frame1=cv2.Canny(frame1,10,200)
    #con
        contour,h=cv2.findContours(frame1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #print(contour)
        #contour = contour[0] if len(contour) == 2 else contour[1]
        #contour=imutils.grab_contours(contour)

        #contour = sorted(contour, key = cv2.contourArea, reverse = True)[:10]
        #print(contour)
        #screenCnt = 0
        rect=np.zeros((4,2))
        for component in zip(contour,h[0]):

            c=component[0]
            hy=component[1]
            perimeter=cv2.arcLength(c,True)
            #print(hy)

            approx=cv2.approxPolyDP(c,0.09*perimeter,True)
            #print(1)
            if cv2.contourArea(c)>800 and len(approx) == 4 and hy[3]==-1:
                #print(cv2.contourArea(c))
                screenCnt = approx
                #print(screenCnt)

                #remove_mask=np.ones((n,m),np.uint8)
                cv2.drawContours(frame5,[screenCnt],-1,(0,255,0), thickness=cv2.FILLED )
                cv2.fillPoly(frame2,[screenCnt],(0,255,0))
                #croped=img[]
                #toplleftx=

                rect_cut=cv2.boundingRect(screenCnt)
                x,y,w,h=rect_cut
                croped = frame3[y:y+h, x:x+w].copy()


                cv2.drawContours(frame2,[screenCnt],-1,(0,255,0),3)
                screenCnt=screenCnt.reshape(4,2)
                rect=np.zeros((4,2))
                sum_c=screenCnt.sum(axis=1)
                counter=1
                rect[0]=screenCnt[np.argmin(sum_c)]
                rect[2]=screenCnt[np.argmax(sum_c)]
                diff_c=np.diff(screenCnt,axis=1)
                rect[1]=screenCnt[np.argmin(diff_c)]
                rect[3]=screenCnt[np.argmax(diff_c)]
                #print("point_value",cv2.pointPolygonTest(rect, (351,247), True))

                break

        #cv2.bitw
        #print(counter)
        extraction_1=cv2.bitwise_xor(frame3,frame2)
        #cv2.imshow('extraction_1',extraction_1)

        #cv2.imshow("mask",frame2)



        #cv2.imshow("edges",edges)

        #plt.imshow(img_back)
        #cv2.imshow("contour",frame2)
        #cv2.imshow("magnitude",img_back)
        #cv2.imshow("final_img",frame1)
        #cv2.imshow('croped',croped)
        #print("42",screenCnt)
        #print(rect)
        return rect,frame3,screenCnt,extraction_1,counter,frame4
        #return rect,croped

    def homography(self,src_pts,dest_pts):


        """

            This function calculates the homography matrix for image_transformation
            from source image to destination image .

            Parameters
            ----------
            src_pts : np.array
                It is the corner points/ the vertices of the source image
                from which we need to transform
            dest_pts : np.array
                It is the corner points/ the vertices of the dest image
                to which we need to transform

            Returns
            -------
            H : np.array
                Homography/transformation matrix


        """
        x=src_pts[:,0]
        y=src_pts[:,1]
        xp=dest_pts[:,0]
        yp=dest_pts[:,1]

        A=np.zeros((8,9))

        for i in range(4):
            A[2*i,0]=-x[i]
            A[2*i,1]=-y[i]
            A[2*i,2]= -1
            A[2*i+1,3]=-x[i]
            A[2*i+1,4]=-y[i]
            A[2*i+1,5]= -1
            A[2*i,6]=x[i]*xp[i]
            A[2*i,7]=y[i]*xp[i]
            A[2*i,8]= 1*xp[i]
            A[2*i+1,6]=x[i]*yp[i]
            A[2*i+1,7]=y[i]*yp[i]
            A[2*i+1,8]= 1*yp[i]



        U,S,V=np.linalg.svd(A)

        H=V[8].reshape((3,3))


        H=(1/H[2,2])*H
        return H
    def image_transformation(self,image,frame_copy,H):
        """

            This function takes the homography gives an perspective transform
            source image(testudo) to destination image(video frame)

            Parameters
            ----------
            image : np.array
                source image which need to be transformed
            frame_copy : np.array
                destination image i.e. the video frame
            H  : np.array
                Homography matrix calculated from source image to destination image

            Returns
            -------
            frame_copy : np.array
                Video Frame/Image after perspective transgorm


        """
        m,n,color=image.shape
        H_inv=np.linalg.inv(H)

        for i in range(m):
            for j in range(n):
                    world_coordinates=np.array([i,j,1])

                    X=np.abs(H @ world_coordinates)

                    X=X/X[2]

                    frame_copy[int(X[1])][int(X[0])]=image[i][j]

        frame_copy=frame_copy.astype(np.uint8)

        return frame_copy


    def wrapping(self,frame_ar,H,src_pts):
        """

            Function takes the AR tag from the image and applys Perspective
            transform and makes it straight for easy dteection of AR tag

            Parameters
            ----------
            frame_ar : np.array
                Frame in which ar tag is present
            H : np.array
                Homography matrix for ar tag in the image  and the AR tag
                in straight orientation
            src_pts : np.array
                    Corners of AR tag

            Returns
            -------
            img : np.array
                straight AR tag/ inverse warped perspective image


        """
        img=np.zeros((80,80))

        H_inv=np.linalg.inv(H)
        #m=
        #=frame_ar.shape
        for i in range(80):
            for j in range(80):



                    world_coordinates=np.array([i,j,1])
                    #X=H@((np.array([i,j,1])).reshape(3,1))
                    X=np.dot(H_inv,world_coordinates)
                    #X=k@X
                    X=X/X[2]
                    if (1080 > X[0] > 0) and (1920 > X[1] > 0):
                    #print(X[0])
                        img[j][i]=frame_ar[int(X[1])][int(X[0])]

        #AR_tag= H @ frame
        img=img.astype(np.uint8)
        #cv2.imshow("AR_tag",img)
        return img
    def orientation(self,img):
        """

            This function takes the prespective transformed image of the ar tag
            and uses it to calculate the rotation and ID of the tag
            and return the orientation which is used further for rotating the
            testudo

            Parameters
            ----------
            img : np.array
                Perspective transformed image of the AR tag

            Returns
            -------
            Index : Int
                Gives the index depending upon which the orientation is decoded
                e.g. index=3 rotation angle 0
                index=4 rotation angle 90


        """
        m ,n =img.shape


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

        elif index == 1:
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

        elif index == 2:
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

        elif index == 3:
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

        return index

    def cube(self,frame,H,cube_pts):
        """

        This function takes the homography and uses it to calculate the rotation
        and translation of the cube with repect to image tag. Transforms and plot
        the cube on the images

        Parameters
        ----------
        frame : np.array
            Frame of the video on which the cube needs to be drawn
        H : np.array
            Homography matrix for cube points and the AR tag1
        cube_pts : np.array
            8 vertices of cube

        Returns
        -------
        frame : np.array
            Final image after plotting the cube


        """

        K = np.array(
            [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T
        #H = inv(H)
        P = (np.linalg.inv(K) @ H)
        P1 = P[:, 0]
        P2 = P[:, 1]
        P3 = P[:,2]
        lamda=math.sqrt(np.linalg.norm(P1,2)*np.linalg.norm(P2,2))

        r1=P1/lamda
        r2=P2/lamda
        trans=P3/lamda
        c=r1+r2
        r3=np.cross(r1,r2)
        d=np.cross(c,r3)

        r1 = np.dot(c / np.linalg.norm(c, 2) + d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        r2 = np.dot(c / np.linalg.norm(c, 2) - d / np.linalg.norm(d, 2), 1 / math.sqrt(2))
        r3=np.cross(r1,r2)






        R = np.stack((r1, r2, r3,trans)).T
        P_P = K @ R
        pts=P_P @ cube_pts.T
        #print(pts.shape)
        x1,y1,z1 = pts[:,0]
        x2,y2,z2 = pts[:,1]
        x3,y3,z3 = pts[:,2]
        x4,y4,z4 = pts[:,3]
        x5,y5,z5 = pts[:,4]
        x6,y6,z6 = pts[:,5]
        x7,y7,z7 = pts[:,6]
        x8,y8,z8 = pts[:,7]


        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x5/z5),int(y5/z5)), (255,0,0), 2)
        cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x6/z6),int(y6/z6)), (255,0,0), 2)
        cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x7/z7),int(y7/z7)), (255,0,0), 2)
        cv2.line(frame,(int(x4/z4),int(y4/z4)),(int(x8/z8),int(y8/z8)), (255,0,0), 2)

        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x2/z2),int(y2/z2)), (0,255,0), 2)
        cv2.line(frame,(int(x1/z1),int(y1/z1)),(int(x3/z3),int(y3/z3)), (0,255,0), 2)
        cv2.line(frame,(int(x2/z2),int(y2/z2)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)
        cv2.line(frame,(int(x3/z3),int(y3/z3)),(int(x4/z4),int(y4/z4)), (0,255,0), 2)

        cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x6/z6),int(y6/z6)), (0,0,255), 2)
        cv2.line(frame,(int(x5/z5),int(y5/z5)),(int(x7/z7),int(y7/z7)), (0,0,255), 2)
        cv2.line(frame,(int(x6/z6),int(y6/z6)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
        cv2.line(frame,(int(x7/z7),int(y7/z7)),(int(x8/z8),int(y8/z8)), (0,0,255), 2)
        cv2.imshow("cube", frame)
        return frame
        #return r, t, K
