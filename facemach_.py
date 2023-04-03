import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpfacemash = mp.solutions.mediapipe.python.solutions.face_mesh
mpconnection = mp.solutions.mediapipe.python.solutions.face_mesh_connections
facemesh = mpfacemash.FaceMesh()
drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
drawspec = drawing.DrawingSpec(thickness=1,circle_radius=1,color=(0, 500, 0))

ptime = 0
while True:
    success , img = cap.read()
    img = cv2.flip(img,1)

    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = facemesh.process(imgRGB)

    if results.multi_face_landmarks :
        for faceLms in results.multi_face_landmarks:
            for id,lm in enumerate(faceLms.landmark):
                h,w,c = img.shape
                x, y = int(lm.x * w), int(lm.y * h)
                # print(id,x,y)
            drawing.draw_landmarks(img,faceLms,mpfacemash.FACEMESH_TESSELATION,drawspec,drawspec)


    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
