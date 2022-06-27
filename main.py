import numpy as np
import cv2
import math

# To find our angle, I think that we can use a process like this:
# 1. Find pixel position of center (cx)
# 2. Find pixel position of nearest numbered tick (nx), as well as number it(N)
# 3. Find width in pixels of distance between small ticks
# 4. divide distance in pixels by small tick ratio and add to N
# Ex for test2.jpg: 19.5 + ((378-191)/10)*0.01 = 19.68

class SpectrometerAngleEstimator(object):    
    @staticmethod
    def from_frame(source):
        frame = cv2.imread(source, 0) #TODO video support
        x_mid = frame.shape[1]/2

        while True:
            img = frame

            lsd = cv2.createLineSegmentDetector(0)
            lines = lsd.detect(img)[0]

            #draw all vertical line segments
            segments = []
            true_mid = lines[0][0][0]
            for i in range(0, len(lines)):
                l = lines[i]
                if abs(l[0][0] - l[0][2]) < abs(l[0][1] - l[0][3]):
                    segments.append(l)
                    if abs(x_mid - l[0][0]) < abs(x_mid - true_mid):
                        true_mid = l[0][0]

            final = lsd.drawSegments(img, np.asarray(segments))    

            cv2.imshow("Detector", final)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = SpectrometerAngleEstimator.from_frame("test2.jpg")
