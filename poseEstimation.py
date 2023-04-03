import cv2
import mediapipe as mp
import time 


mypose = mp.solutions.mediapipe.python.solutions.pose
pose = mypose.Pose()
drawing = mp.solutions.mediapipe.python.solutions.drawing_utils

cap  = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FPS,30)

ptime = 0
while True:
    
    success, img = cap.read()
    img = cv2. resize(img,(800,600))
    ctime= time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,222,0),3)
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    if results.pose_landmarks:
        
        drawing.draw_landmarks(img,results.pose_landmarks,mypose.POSE_CONNECTIONS)
        for id , lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy =int(w * lm.x),int(h * lm.x)
            print(id,cx,cy)

    

    cv2.imshow("video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break