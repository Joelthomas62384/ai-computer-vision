import cv2
import face_mesh_module as fmm
import time


cap = cv2.VideoCapture(0)
face = fmm.faceMesh_detection()

while True:
    _,img = cap.read()
    img = cv2.flip(img,2)
    img = face.find_mesh(img)
    cv2.imshow("video",img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

