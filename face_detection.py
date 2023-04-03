import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFace = mp.solutions.mediapipe.python.solutions.face_detection
face = mpFace.FaceDetection()
drawing = mp.solutions.mediapipe.python.solutions.drawing_utils


ptime = 0
ctime = 0 
while True:
    success , img = cap.read()
    img = cv2.flip(img,1)
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime

    cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = face.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
            h , w ,c = img.shape
            cbox = detection.location_data.relative_bounding_box
            bbox = int(cbox.xmin * w), int(cbox.ymin *h), int(cbox.width *w), int(cbox.height * h)
            cv2.rectangle(img,bbox,(255,0,255),3)
            # print(cbox)
            
            # print(score_len)
            for i in range(len(detection.score)):
                cv2.putText(img,f"{int(detection.score[i] * 100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)


    cv2.imshow("Video",img)

    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
