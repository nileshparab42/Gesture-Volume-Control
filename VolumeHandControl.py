import cv2 
import mediapipe as mp
import time
import numpy as np
import math
import HandTrackingModule as htm
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(0)
ptime = 0
ctime = 0
detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))


volRange = volume.GetVolumeRange()   #  (-65.25, 0.0, 0.03125)
minVol = volRange[0]
maxVol = volRange[1]
volBar = 400
volPer = 0

while True:
    success , img = cap.read()
    img= detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList)!=0:
        # print(lmList[5])
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx,cy = (x1+x2)//2,(y1+y2)//2

        cv2.circle(img,(x1,y1),10,(255,0,255),cv2.FILLED)
        cv2.circle(img,(x2,y2),10,(255,0,255),cv2.FILLED)
        cv2.line(img,(x1,y1),(x2,y2),(255,0,255),3)
        cv2.circle(img,(cx,cy),10,(255,0,255),cv2.FILLED)

        length = math.hypot(x2-x1,y2-y1)        #Euclidean distance
        if length<25:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        if length>160:
            cv2.circle(img,(cx,cy),10,(0,255,0),cv2.FILLED)
        
        vol = np.interp(length,[25,160],[minVol,maxVol])
        volBar = np.interp(length,[25,160],[400,150])
        volPer = np.interp(length,[25,160],[0,100])

        volume.SetMasterVolumeLevel(vol, None)
        
        cv2.rectangle(img,(50,150),(85,400),(0,255,0),3)
        cv2.rectangle(img,(50,int(volBar)),(85,400),(0,255,0),cv2.FILLED)
        cv2.putText(img,f'{int(volPer)} %',(40,450),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)

    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime=ctime
    cv2.putText(img,f'FPS: {int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)