import cv2
import numpy as np

class HSVAdjustmentGUI:
    def __init__(self):
        self.capture = cv2.VideoCapture(0)
        self.window_name = "HSV Adjustment"
        cv2.namedWindow(self.window_name)

        # Create trackbars for HSV values
        cv2.createTrackbar("HMin", self.window_name, 0, 179, self.update_image)
        cv2.createTrackbar("HMax", self.window_name, 179, 179, self.update_image)
        cv2.createTrackbar("SMin", self.window_name, 0, 255, self.update_image)
        cv2.createTrackbar("SMax", self.window_name, 255, 255, self.update_image)
        cv2.createTrackbar("VMin", self.window_name, 0, 255, self.update_image)
        cv2.createTrackbar("VMax", self.window_name, 255, 255, self.update_image)

        self.update_image()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        self.capture.release()
        cv2.destroyAllWindows()

    def update_image(self, *args):
        _, frame = self.capture.read()

        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current trackbar values
        hmin = cv2.getTrackbarPos("HMin", self.window_name)
        hmax = cv2.getTrackbarPos("HMax", self.window_name)
        smin = cv2.getTrackbarPos("SMin", self.window_name)
        smax = cv2.getTrackbarPos("SMax", self.window_name)
        vmin = cv2.getTrackbarPos("VMin", self.window_name)
        vmax = cv2.getTrackbarPos("VMax", self.window_name)

        # Set lower and upper bounds for the HSV values
        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])

        # Threshold the HSV image to get only the desired colors
        mask = cv2.inRange(hsv, lower, upper)

        # Show the original and thresholded images side by side
        cv2.imshow(self.window_name, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))

HSVAdjustmentGUI()
