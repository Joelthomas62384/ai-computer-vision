import cv2
import time
import os
import hand_tracker as ht

wcam,hcam = 640,480

cap = cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)
ctime = 0
ptime = 0
detector = ht.hand_detector(detection_confidence=0.75)
fngrids = [4,8,12,16,20]
while True:
    success,img = cap.read()
    img = cv2.flip(img,2)
    img = detector.hand_detection(img,draw=True)

    lmlist = detector.findposition(img)
    # print(lmlist)
    fingers = []
    if len(lmlist) != 0:
        if lmlist[fngrids[0]][1] < lmlist[fngrids[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        for id in range(1,5):
            if lmlist[fngrids[id]][2] < lmlist[fngrids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

    # print(fingers)
        totalfingers = fingers.count(1)
        print(totalfingers)
        cv2.rectangle(img,(20,255),(170,445),(0,255,15),cv2.FILLED)
        cv2.putText(img,str(totalfingers),(45,400),cv2.FONT_HERSHEY_DUPLEX,5,(255,0,0),15)






    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,f"FPS : {int(fps)}",(10,70),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)

    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break