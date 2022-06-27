import numpy as np
import cv2
import math

class SpectrometerAngleEstimator(object):
    def __init__(self, source):
        self.frame = cv2.imread(source, 0) #TODO video support

    def angle_from_frame(self):
        while True:
            img = self.frame

            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(img)[0]

            print(type(lines[0]))
            segments = []
            for i in range(0, len(lines)):
                l = lines[i]
                if abs(l[0][0] - l[0][2]) < abs(l[0][1] - l[0][3]):
                    segments.append(lines[i])

            img = lsd.drawSegments(img, np.asarray(segments))

            cv2.imshow("Detector", img)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = SpectrometerAngleEstimator("test2.jpg")
    model.angle_from_frame()
