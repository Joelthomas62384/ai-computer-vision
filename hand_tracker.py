import cv2
import mediapipe as mp
import time
import numpy as np


class hand_detector:
    def __init__(self,mode = False,max_hands=2,model_complexity=1,detection_confidence=0.5,tracking_confidence=0.5) :
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.tipids = [4,8,12,16,20]
        


        self.mphands = mp.solutions.mediapipe.python.solutions.hands
        self.hands = self.mphands.Hands(self.mode,self.max_hands,self.model_complexity,self.detection_confidence,self.tracking_confidence)
        self.mpdraw = mp.solutions.mediapipe.python.solutions.drawing_utils
        

    def hand_detection(self,img,draw = False):
      
        imgRgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRgb)
        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img,handlms,self.mphands.HAND_CONNECTIONS)
        

        return img

    def findposition(self,img,handno = 0,draw=False):
        
            self.lmlist = []
            if self.results.multi_hand_landmarks:
                myhand = self.results.multi_hand_landmarks[handno]
                for id,lm in enumerate(myhand.landmark):
                    h,w,c = img.shape
                    cx , cy = int(lm.x*w),int(lm.y*h)
                    self.lmlist.append([id,cx,cy])
                    if draw:
                        cv2.circle(img,(cx,cy),3,(200,244,0),3)

            return self.lmlist
    def finguresUp(self):
        fingers = []
        if self.lmlist[self.tipids[0]][1] < self.lmlist[self.tipids[0]-1][1]:
                fingers.append(1)
        else:
                fingers.append(0)
        for id in range(1,5):
            if self.lmlist[self.tipids[id]][2] < self.lmlist[self.tipids[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers
    def findDistance(self,img):
        x , y = self.lmlist[12][1] , self.lmlist[12][2]
        x1 , y1 = self.lmlist[8][1], self.lmlist[8][2]
        cx, cy = (x+x1)//2,(y+y1)//2
        length = np.hypot(x1 - x,y1-y)
        cv2.circle(img,(x , y),8,(0,255,0),-1)
        cv2.circle(img,(x1 , y1),8,(0,255,0),-1)
        cv2.circle(img,(cx , cy),8,(0,255,0),-1)
        if length < 65:
            cv2.circle(img,(cx , cy),8,(0,0,255),-1)
        
        cv2.line(img,(x,y),(x1,y1),(0,255,0),2)
        return length
        


def main():

    cap = cv2.VideoCapture(0)
    ctime = 0
    ptime = 0
    detector = hand_detector()
    while True:
        success , img = cap.read()
        img = cv2.flip(img,2)
        img = detector.hand_detection(img,draw = True)
        lmlist = detector.findposition(img,draw=True)
        if len(lmlist) != 0:
            print(lmlist[0])
        



        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        cv2.putText(img,f"FPS: {int(fps)}",(10,70),cv2.FONT_HERSHEY_COMPLEX,2,(255,255,255),5)
        cv2.imshow('video',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()