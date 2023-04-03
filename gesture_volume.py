import cv2
import mediapipe as mp
import time
import numpy as np
import hand_tracker as ht


from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

volume.GetVolumeRange()
volRnge = volume.GetVolumeRange()
maxvol =volRnge[1]
minvol = volRnge[0]

cap = cv2.VideoCapture(0)
print(minvol)
ptime = 0
volbar = -65.25
detector = ht.hand_detector(detection_confidence=0.75)
while True:
    success, img = cap.read()
    img = cv2.flip(img,1)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,f"FPS : {int(fps)}",(10,80),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
    detector.hand_detection(img)
    lmlis = detector.findposition(img,draw = False)
    if len(lmlis) != 0:

        x , y = lmlis[4][1] , lmlis[4][2]
        x1 , y1 = lmlis[8][1], lmlis[8][2]
        cx, cy = (x+x1)//2,(y+y1)//2
        length = np.hypot(x1 - x,y1-y)
        vol = np.interp(length,[30,110],[minvol,maxvol])
        volbar = np.interp(vol,[minvol,maxvol],[400,150])
        volper = np.interp(vol,[minvol,maxvol],[0,100])
        volume.SetMasterVolumeLevel(vol, None)

        cv2.circle(img,(x , y),8,(0,255,0),-1)
        cv2.circle(img,(x1 , y1),8,(0,255,0),-1)
        cv2.circle(img,(cx , cy),8,(0,255,0),-1)
        
        cv2.line(img,(x,y),(x1,y1),(0,255,0),2)
        cv2.rectangle(img,(150,150),(85,400),(0,125,0))
        cv2.rectangle(img,(150,int(volbar)),(85,400),(0,125,0),-1)
        cv2.putText(img,f"{int(volper)}%",(30,150),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)

        if length < 35:
            cv2.circle(img,(cx , cy),8,(0,0,255),-1)

        








    cv2.imshow("Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    

    