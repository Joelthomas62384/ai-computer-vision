import pyautogui as pg
import cv2
import hand_tracker as ht
import time
import numpy as np

wscr ,hscr = pg.size()
smoothening =5
 
plocx , plocy = 0,0
clocx , clocy = 0,0

cap = cv2.VideoCapture(0)

framer = 150

wcam = 1280
hcam = 780
cap.set(3,wcam)
cap.set(4,hcam)
hands = ht.hand_detector(detection_confidence=0.60)
ctime = 0
ptime = 0
while True:
    success , img = cap.read()
    img = cv2.flip(img,2)
    img = hands.hand_detection(img,draw=True)
    handlms = hands.findposition(img)
    cv2.rectangle(img,(framer,framer),(wcam - framer,hcam-framer),(255,0,255),2)
    if len(handlms) != 0:
        x1,y1 =handlms[8][1:]
        x2,y2 =handlms[12][1:]
        

        fingers = hands.finguresUp()
        if fingers[1] and fingers[2] == False:
            
            x3 = np.interp(x1,(framer,wcam-framer),(0,wscr))
            y3 = np.interp(y1,(framer,hcam-framer),(0,hscr))
            clocx = plocx+(x3-plocx)/smoothening
            clocy = plocy+(y3-plocy)/smoothening
            cv2.circle(img,(x1,y1),15,(0,255,0),cv2.FILLED)
            
            pg.moveTo(clocx,clocy)
            plocx,plocy = clocx,clocy
        elif fingers[1] and fingers[2]:
            length = hands.findDistance(img)
            if length<60:
                pg.click(clocx,clocy)






    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime


    cv2.putText(img,f'FPS : {int(fps)}',(0,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),2)
    cv2.imshow("Mouse Controller",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

