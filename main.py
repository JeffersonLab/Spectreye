import numpy as np
import cv2
import math
import sys
from imutils.object_detection import non_max_suppression
import pytesseract

# To find our angle, I think that we can use a process like this:
# 1. Find pixel position of center (cx)
# 2. Find pixel position of nearest numbered tick (nx), as well as number it(N)
# 3. Find width in pixels of distance between small ticks
# 4. divide distance in pixels by small tick ratio and add to N
# Ex for test2.jpg: 19.5 + ((378-191)/10)*0.01 = 19.68

layer_names = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

PIXEL_RATIO = 10    #accurate estimation for now
NPAD = 15
MKERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
FONT = cv2.FONT_HERSHEY_SIMPLEX

#pytesseract.pytesseract.tesseract_cmd = r""

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

            true_mid = 378 #temp

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
                    
                    if endY*rH < frame.shape[0]/2:
                        rects.append((startX, startY, endX, endY))
                        confidences.append(scores_data[x])

            boxes = non_max_suppression(np.array(rects), probs=confidences)

            cmpX = 0
            boxdata = None
            for(startX, startY, endX, endY) in boxes:
                startX = int(startX * rW)
                startY = int(startY * rH)
                endX = int(endX * rW)
                endY = int(endY * rH)
                cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)
                tpos = startX + (endX-startX)/2
                if abs(true_mid - tpos) < abs(true_mid - cmpX):
                    cmpX = tpos
                    boxdata = [startY-NPAD, endY+NPAD, startX-NPAD, endX+NPAD]
                
            tick = 0
            for l in segments:
                if abs(cmpX - l[0][0]) < abs(cmpX - tick) and l[0][1] > 175 and l[0][1] < 240:
                    tick = l[0][0]
            cv2.rectangle((final), (int(tick), 0), (int(tick), frame.shape[0]), (255, 0, 0), 1)   
           
            pix_frac = true_mid - tick
            dec_frac = (pix_frac/PIXEL_RATIO)*0.01
           
            print("Additional distance (deg): " + str(dec_frac))
           
            #isolate numbox
            numbox = img[boxdata[0]:boxdata[1], boxdata[2]:boxdata[3]]
            numbox = cv2.GaussianBlur(numbox,(5,5),0)
            (_, numbox) = cv2.threshold(numbox,127,255,cv2.THRESH_BINARY_INV)
            numbox = cv2.morphologyEx(numbox, cv2.MORPH_OPEN, MKERNEL)
            rawnum = pytesseract.image_to_string(numbox, lang="eng", config="--psm 6")
            nstr = ""
            print(rawnum)
            for n in rawnum:
                if n.isdigit():
                    nstr += n
            nstr = nstr[:2] + "." + nstr[2:]
            angle = float(nstr) + dec_frac
            print(angle) 

            cv2.putText(final, str(angle), (10, 30), FONT, 1, (0, 255, 0), 2, 2)

            #display and poll
            cv2.imshow("Detector", final)
#            cv2.imshow("Numbers", numbox)
            while True:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            break
        cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        SpectrometerAngleEstimator.from_frame(sys.argv[1])
    else:
        SpectrometerAngleEstimator.from_frame("test2.jpg")
