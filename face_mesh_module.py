import cv2
import mediapipe as mp
import time


#  static_image_mode=False,
#                max_num_faces=1,
#                refine_landmarks=False,
#                min_detection_confidence=0.5,
#                min_tracking_confidence=0.5


class faceMesh_detection():
    def __init__ (self,mode=False,refineLms = False, detection_conf = 0.5, tracking_conf = 0.5):
        self.mode = mode
        self.refineLms = refineLms
        self.detection_conf = detection_conf
        self.tracking_conf = tracking_conf

        self.mpfacemash = mp.solutions.mediapipe.python.solutions.face_mesh
        self.facemesh = self.mpfacemash.FaceMesh()
        self.drawing = mp.solutions.mediapipe.python.solutions.drawing_utils
        self.drawspec = self.drawing.DrawingSpec(thickness=1,circle_radius=1,color=(0, 500, 0))

    def find_mesh(self,img,draw=True):


        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

        results = self.facemesh.process(imgRGB)
        faces = []
        if results.multi_face_landmarks :
            face = []
            for faceLms in results.multi_face_landmarks:
                for id,lm in enumerate(faceLms.landmark):
                    h,w,c = img.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    # print(id,x,y)
                    face.append([x,y])
                faces.append(face)
                if draw:
                    self.drawing.draw_landmarks(img,faceLms,self.mpfacemash.FACEMESH_TESSELATION,self.drawspec,self.drawspec)

        return img,faces


def main():
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = faceMesh_detection()
    while True:
        success , img = cap.read()
        img = cv2.flip(img,1)
        ctime = time.time()
        fps = 1/(ctime - ptime)
        ptime = ctime
        img,faces = detector.find_mesh(img)
        if len(faces) != 0:
            print(faces)
        cv2.putText(img,str(int(fps)),(10,80),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),3)
        cv2.imshow("Video",img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()