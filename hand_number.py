import cv2
import time
import os


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

folderpath  = "hand_number"
imglis = os.listdir(folderpath)
print(imglis)
overlay = []
for imgpath in imglis:
    image = cv2.imread(f"{folderpath}/{imgpath}")
    # print(f"{folderpath}/{imgpath}")
    overlay.append(image)

    # print(len(overlay))

while True:
    success, img = cap.read()
    img = cv2.flip(img,1)

    img[1080:1000,480:640] = overlay[1]



    cv2.imshow("Video",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break