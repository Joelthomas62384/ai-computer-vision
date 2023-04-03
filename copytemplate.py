import cv2
import time
import hand_tracker as htm



cap = cv2.VideoCapture(0)
ptime = 0
while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime 

    cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,222,0),3)
    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
