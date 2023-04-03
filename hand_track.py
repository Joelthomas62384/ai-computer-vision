import cv2
import mediapipe as mp
import time


mphands = mp.solutions.mediapipe.python.solutions.hands
hands = mphands.Hands()
mpdraw = mp.solutions.mediapipe.python.solutions.drawing_utils



cap = cv2.VideoCapture(0)

ctime = 0
ptime = 0
while True:
    success , img = cap.read()
    img = cv2.flip(img,2)

    imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRgb)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            for id,lm in enumerate(handlms.landmark):
                mpdraw.draw_landmarks(img,handlms,mphands.HAND_CONNECTIONS)
                h,w,c = img.shape
                cx , cy = int(lm.x*w),int(lm.y*h)
                print(id,cx,cy)




    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,f"FPS: {int(fps)}",(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
    cv2.imshow('video',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
