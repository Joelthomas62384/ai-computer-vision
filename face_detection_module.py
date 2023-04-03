import cv2
import mediapipe as mp
import time

class Face_detection():
    # min_detection_confidence=0.5, model_selection=0
    def __init__(self,detection_conf = 0.5,model_select=0):
        self.detection_conf = detection_conf
        self.model_select = model_select


        

        self.mpFace = mp.solutions.mediapipe.python.solutions.face_detection
        self.face = self.mpFace.FaceDetection(self.detection_conf,self.model_select)
        self.drawing = mp.solutions.mediapipe.python.solutions.drawing_utils


    def findFace(self,img,draw = True):
        
        bboxs = []

        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = self.face.process(imgRGB)
        if results.detections:
            for id,detection in enumerate(results.detections):
                h , w ,c = img.shape
                cbox = detection.location_data.relative_bounding_box
                bbox = int(cbox.xmin * w), int(cbox.ymin *h), int(cbox.width *w), int(cbox.height * h)
                # print(bbox)
                bboxs.append([id,bbox,detection.score])
                if draw:    
                    self.draw_box(img,bbox)
                    # print(score_len)
                    for i in range(len(detection.score)):
                        cv2.putText(img,f"{int(detection.score[i] * 100)}%",(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        return img, bboxs

    def draw_box(self,img,bbox,l=30):
        # print(bbox)
        x, y , w , h = bbox
        x1 ,y1 = x+w,y+h
        cv2.rectangle(img,bbox,(255,0,255),1)
        

        cv2.line(img,(x,y),(x+l,y),(255,0,255),4)
        cv2.line(img,(x,y),(x,y+l),(255,0,255),4)

        cv2.line(img,(x1,y),(x1-l,y),(255,0,255),4)
        cv2.line(img,(x1,y),(x1,y+l),(255,0,255),4)

        cv2.line(img,(x,y1),(x+l,y1),(255,0,255),4)
        cv2.line(img,(x,y1),(x,y1-l),(255,0,255),4)

        cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),4)
        cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),4)

        # cv2.line(img,)

   


def main():
    ptime = 0
    ctime = 0 
    cap = cv2.VideoCapture(0)
    detector = Face_detection()
    while True:
        success , img = cap.read()
        img = cv2.flip(img,1)
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime

        img,bboxs = detector.findFace(img)
        # print(bboxs)

        cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
        
        cv2.imshow("Video",img)

        if cv2.waitKey(1) & 0xFF ==ord('q'):
            break


if __name__ == "__main__":
    main()