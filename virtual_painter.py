import cv2
import time
import numpy as np
import os
import hand_tracker as htm
import face_mesh_module as fmm


brushthickness = 10
eraserthickness = 50
folderpath = 'header'
mylist = os.listdir(folderpath)
# print(mylist)
overlaylist = []
for imagepath in mylist:
    image = cv2.imread(f'{folderpath}/{imagepath}')
    overlaylist.append(image)


header = overlaylist[0]
drawcolor = (255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,750)
ctime = 0
ptime = 0

detector = htm.hand_detector(detection_confidence=0.8)
face = fmm.faceMesh_detection(detection_conf=0.75)
xp , yp =0,0
imgcanvas = np.zeros((750,1280,3),np.uint8)


while True:
    # 1. Import Image
    success , img = cap.read()
    img = cv2.flip(img,2)
    img = cv2.resize(img,(1280,750))
    # print(img.shape)
    # 2. Find Landmarks
    img , faces = face.find_mesh(img)
    
    img = detector.hand_detection(img,True)
    

    # tip finder

    lmlist = detector.findposition(img)
    if len(lmlist) !=0:
        x1 , y1 = lmlist[8][1:]
        x2 , y2 = lmlist[12][1:]

        fingers = detector.finguresUp()
        # print(fingers)

        # 3. selction mode finder
        if fingers[1] and fingers[2]:
            cv2.rectangle(img,(x1,y1-35),(x2,y2+35),drawcolor,cv2.FILLED)
            print("selection mode")
            xp , yp =0,0


            if y1 <135:
                if 250<x1<400:
                    header = overlaylist[0]
                    drawcolor = (255,0,255)
                    

                elif 430<x1<570:
                    header = overlaylist[1]
                    drawcolor = (255,0,0)
                elif 670<x1<800:
                    header = overlaylist[2]
                    drawcolor = (0,255,0)

                elif 900<x1<1000:
                    header = overlaylist[3]
                    drawcolor = (0,0,0)

        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),10,drawcolor,cv2.FILLED)
            print("Drawing mode")
            if xp ==0 and yp ==0:
                xp , yp = x1, y1
            if drawcolor == (0,0,0):
                cv2.line(img,(xp,yp),(x1,y1),drawcolor,eraserthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,eraserthickness)
            else:

                cv2.line(img,(xp,yp),(x1,y1),drawcolor,brushthickness)
                cv2.line(imgcanvas,(xp,yp),(x1,y1),drawcolor,brushthickness)
            xp,yp = x1,y1


   
    imgGray = cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY )
    _,imginv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
   
    imginv = cv2.cvtColor(imginv,cv2.COLOR_GRAY2BGR)
    # 
    # imgcanvas = cv2.resize(imgcanvas,(1280,720))
    # print(f"img {img.shape}, imginv {imginv.shape}")
    img = cv2.bitwise_and(img,imginv)
    img = cv2.bitwise_or(img,imgcanvas)




    img[0:135,0:1280] = header

    cv2.imshow('Virtual painter by Joel Thomas',img)
    # cv2.imshow(' Joel Thomas',imgface)
    # cv2.imshow('canvas',imgcanvas)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

    
