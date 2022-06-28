import numpy as np
import cv2
import math
from imutils.object_detection import non_max_suppression


# To find our angle, I think that we can use a process like this:
# 1. Find pixel position of center (cx)
# 2. Find pixel position of nearest numbered tick (nx), as well as number it(N)
# 3. Find width in pixels of distance between small ticks
# 4. divide distance in pixels by small tick ratio and add to N
# Ex for test2.jpg: 19.5 + ((378-191)/10)*0.01 = 19.68

layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

class SpectrometerAngleEstimator(object):    
    @staticmethod
    def from_frame(source):
        frame = cv2.imread(source) #TODO video support
        x_mid = frame.shape[1]/2

        while True:         #for now having a loop is useless but will be necessary for video
            img = frame 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

            #detect text
            timg = frame
            if len(timg.shape) == 2:
                timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2RGB)
            (H, W) = timg.shape[:2]
            (newW, newH) = (320, 320)
            rW = W / float(newW)
            rH = H / float(newH)

            timg = cv2.resize(timg, (newW, newH))
            (H, W) = timg.shape[:2]

            net = cv2.dnn.readNet("east.pb")
            blob = cv2.dnn.blobFromImage(timg, 1.0, (W, H),
                    (123.68, 116.78, 103.94), swapRB=True, crop=False) #dont touch magic nums
            net.setInput(blob)
            (scores, geometry) = net.forward(layer_names)
            (nrows, ncols) = scores.shape[2:4]
            rects = []
            confidences = []
            for y in range(0, nrows):
                scores_data = scores[0, 0, y]
                xdata0 = geometry[0, 0, y]
                xdata1 = geometry[0, 1, y]
                xdata2 = geometry[0, 2, y]
                xdata3 = geometry[0, 3, y]
                angles = geometry[0, 4, y]

                for x in range(0, ncols):
                    if scores_data[x] < 0.5:
                        continue
                    (offsetX, offsetY) = (x * 4.0, y * 4.0)
                    angle = angles[x]
                    cos = np.cos(angle)
                    sin = np.sin(angle)
                    h = xdata0[x] + xdata2[x]
                    w = xdata1[x] + xdata3[x]

                    endX = int(offsetX + (cos * xdata1[x]) + (sin * xdata2[x]))
                    endY = int(offsetY - (sin * xdata1[x]) + (cos * xdata2[x]))
                    startX = int(endX - w)
                    startY = int(endY - h)

                    if startY < frame.shape[0]/2:
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scores_data[x])

            boxes = non_max_suppression(np.array(rects), probs=confidences)

            cmpX = 0
            for(startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)
                tpos = startX + (endX-startX)/2
                if abs(true_mid - tpos) < abs(true_mid - cmpX):
                    cmpX = tpos
                
            tick = 0
            for l in segments:
                if abs(cmpX - l[0][0]) < abs(cmpX - tick) and l[0][0] > 175 and l[0][0] < 240:
                    tick = l[0][0]

            cv2.rectangle((final), (int(tick), 0), (int(tick), frame.shape[0]), (255, 0, 0), 1)
            
            #display and poll
            cv2.imshow("Detector", final)
            while True:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    model = SpectrometerAngleEstimator.from_frame("test2.jpg")
