import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft,ifft
import scipy
import imutils

import Tag_detection_tracking

import os

def detecting_tag(frame):
      """
      Takes the fourier transformation of the image to remove the noise from the image
      and then inverse fourier transformation is taken so that which gives the output image
      which is then used to extract AR tag

      Parameters
      ----------
      frame : np.array /img
          Input image from the video


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
      frame5=frame.copy()
      frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      frame1=frame.copy()
      frame2=frame.copy()
      frame3=frame.copy()

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

      fshift = fshift*mask
      #fshift=cv2.bitwise_and(fshift,mask)
      #fshift[crow-20:crowq+20, ccol-20:ccol+20] = 0
      f_ishift = scipy.fft.ifftshift(fshift)

      img_back = scipy.fft.ifft2(f_ishift)
      img_back = (np.abs(img_back))
      img_back=img_back.astype(np.uint8)


      ret,img_back=cv2.threshold(img_back,60,255,cv2.THRESH_BINARY)
      cv2.imshow("after_fourier_transform",img_back)
      frame1=cv2.bitwise_and(frame1,img_back)



      contour,h=cv2.findContours(frame1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
              cv2.drawContours(frame5,[screenCnt],-1,(0,255,0), thickness=2 )
              cv2.fillPoly(frame2,[screenCnt],(0,255,0))
              cv2.imshow("frame5",frame5)
              #croped=img[]
              #toplleftx=


              break

      #cv2.bitw
      #print(counter)
      extraction_1=cv2.bitwise_xor(frame3,frame2)


#FPS_val = 25
out = cv2.VideoWriter('cube.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (1920,1080))

file_location=os.getcwd()   #gets the current directory
video=cv2.VideoCapture(file_location+"/Tag1.mp4")
testudo=cv2.imread(file_location+"/testudo.png")
test_do1=testudo.copy()
cv2.imshow("testudo_",testudo)
#tag=Tag_detection_tracking.Tag_detection_tracking()


cube_pts=np.array([[0,0,0,1],[0,1,0,1],[1,0,0,1],[1,1,0,1],[0,0,1,1],[0,1,1,1],[1,0,1,1],[1,1,1,1]])

while (video.isOpened()):
    ret,frame=video.read()
    if frame is None:
        break
    detecting_tag(frame)


    if cv2.waitKey(0) & 0xFF==ord('q'):
        break
video.release()
out.release()
cv2.destroyAllWindows()
