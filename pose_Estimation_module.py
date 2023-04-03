import cv2
import mediapipe as mp
import time 


class PoseDetection():
    def __init__(self,mode=False,model_comp=1,smooth_land=True,enable_segmentation=False,smooth_segmentation=True,detect_conf=0.5,track_conf=0.5):
        self.mode = mode
        self.model_comp = model_comp
        self.smooth_land = smooth_land
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detect_conf = detect_conf
        self.track_conf = track_conf

        
        self.mypose = mp.solutions.mediapipe.python.solutions.pose
        self.pose = self.mypose.Pose(self.mode,self.model_comp,self.smooth_land,self.enable_segmentation,self.smooth_segmentation,self.detect_conf,self.track_conf)
        self.drawing = mp.solutions.mediapipe.python.solutions.drawing_utils

    def find_pose(self,img,draw=True):
        self.img = img
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.drawing.draw_landmarks(self.img,self.results.pose_landmarks,self.mypose.POSE_CONNECTIONS)
        return self.img

    def get_position(self,draw=True,):
        lmlis = []
        if self.results.pose_landmarks:
            for id , lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = self.img.shape
                cx,cy =int(w * lm.x),int(h * lm.x)
                lmlis.append([id,cx,cy])
        return lmlis


def main():
    cap  = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FPS,30)
    detection = PoseDetection()
    ptime = 0
    while True:
        
        success, img = cap.read()
        img = cv2. resize(img,(800,600))
        img=cv2.flip(img,1)
        img = detection.find_pose(img)
        lmlis = detection.get_position()
        if len(lmlis)!=0:
            print(lmlis[4])
        ctime= time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,222,0),3)
        cv2.imshow("video",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    main()