import numpy as np
import cv2

class SpectrometerAngleEstimator(object):
    def __init__(self, source):
        self.frame = cv2.imread(source) #TODO video support

    def angle_from_frame(self):
        while 1:
            img = self.frame

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Detector", img)
            wait = cv2.waitKey(0)
            if wait & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = SpectrometerAngleEstimator("test1.png")
    model.angle_from_frame()
