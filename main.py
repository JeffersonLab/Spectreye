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

class SpectrometerAngleEstimator(object):
    layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

    npad = 100
    font = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.dkernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (4,4))
        self.okernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        self.ckernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))

        self.net = cv2.dnn.readNet("east.pb")
        self.lsd = cv2.createLineSegmentDetector(0)

    def from_frame(self, frame):
        x_mid = frame.shape[1]/2
        y_mid = frame.shape[0]/2

        img = frame 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        (ltick, rtick, segments) = self.get_ticks(img, True)

        true_mid = self.find_mid(img, segments, ltick)[0][0] - (rtick[0][0] - ltick[0][0])
        #true_mid = x_mid

        pixel_ratio = (2 * (rtick[0][0] - ltick[0][0]))
        print("0.01 degrees = " + str(pixel_ratio) + " pixels")
        
#        final = self.lsd.drawSegments(img, np.asarray(segments))
        final = self.lsd.drawSegments(img, np.asarray([ltick, rtick]))
        cv2.rectangle((final), (int(true_mid), 0), (int(true_mid), img.shape[0]), (255, 255, 0), 1)   

        #detect text
        pass1 = img
        pass1 = cv2.threshold(pass1, 127, 255, cv2.THRESH_BINARY_INV, 0)[1]
        bg = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, self.dkernel)
        pass1 = cv2.divide(pass1, bg, scale=255)
        pass1 = cv2.GaussianBlur(pass1, (3, 3), 0)
        pass1 = cv2.morphologyEx(pass1, cv2.MORPH_OPEN, self.okernel)
        pass1 = cv2.GaussianBlur(pass1, (5, 5), 0)
        pass1 = cv2.fastNlMeansDenoising(pass1,None,21,7,21)
        
        timg = pass1
        (boxes, rW, rH) = self.get_boxes_east(timg)

        if len(boxes) == 0:
            cv2.imshow("Failure", timg)
            cv2.waitKey(0)
            raise Exception("Could not locate any numbers!")

        cmpX = 0
        boxdata = None
        for(startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)
            tpos = startX + (endX-startX)/2
            if abs(x_mid - tpos) < abs(x_mid - cmpX):
                cmpX = tpos
                width = abs(endX-startX)
                height = abs(endY-startY)
                boxdata = [
                        max(0, startY-50), 
                        min(frame.shape[0], endY+self.npad), 
                        max(0, startX-self.npad), 
                        min(frame.shape[1], endX+self.npad)
                ]
            
        tick = 0
        for l in segments:
            if abs(cmpX - l[0][0]) < abs(cmpX - tick) and l[0][1] > 175 and l[0][1] < 240:
                tick = l[0][0]
        cv2.rectangle((final), (int(tick), 0), (int(tick), frame.shape[0]), (255, 0, 0), 1)   
       
        pix_frac = true_mid - tick
        dec_frac = (pix_frac/pixel_ratio)*0.01
       
        print("Additional distance (deg): " + str(dec_frac))
       
        #isolate numbox
        numbox = pass1[boxdata[0]:boxdata[1], boxdata[2]:boxdata[3]]
        numbox = cv2.fastNlMeansDenoising(numbox,None,21,7,21)

        rawnum = pytesseract.image_to_string(numbox, lang="eng", config="--psm 6")
        nstr = ""
        print(rawnum)
        for n in rawnum:
            if n.isdigit():
                nstr += n
        if len(nstr) < 3:
            nstr = nstr + ".0"
        else:
            nstr = nstr[:2] + "." + nstr[2:]

        angle = round(float(nstr) + dec_frac, 2)
        print(angle) 

        cv2.putText(final, str(angle), (10, 30), self.font, 1, (0, 255, 0), 2, 2)

        #display and poll
        cv2.imshow("Detector", final)
        #cv2.imshow("Numbers", pass0)
        #cv2.imshow("Box", timg)
        while True:
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    def find_mid(self, img, segments, ltick):
        #iter, if ltick or rtick can be contained find longest
        mid = segments[0]
        x_mid = img.shape[1]/2 

        opts = []
        for l in segments:
            if ltick[0][1] >= l[0][1] and ltick[0][3] <= l[0][3]:
                opts.append(l)
                if l[0][3] - l[0][1] > mid[0][3] - mid[0][1]:
                    mid = l
        cull = []
        for l in opts:
            if not (l[0][1] < ltick[0][1]-(ltick[0][3]-ltick[0][1])):
                cull.append(l)
            elif not (l[0][3] > ltick[0][3]+(ltick[0][3]-ltick[0][1])):
                cull.append(l)
        
        for l in cull:
            if abs(x_mid - l[0][0]) < abs(x_mid - mid[0][0]):
                mid = l

        return mid

    def get_ticks(self, img, debug=False):
        x_mid = img.shape[1]/2
        y_mid = img.shape[0]/2

        pass1 = img
        pass1 = cv2.GaussianBlur(pass1, (5, 5), 0)

        if debug:
            cv2.imshow("Line Filter", pass1)
            cv2.waitKey(0)

        lines = self.lsd.detect(pass1)[0]
        ltick = lines[0]
        rtick = lines[1]

        segments = []
        for i in range(0, len(lines)):
            l = lines[i]
            if abs(l[0][0] - l[0][2]) < abs(l[0][1] - l[0][3]):
                if l[0][1] > y_mid - (0.1 * img.shape[0]) and l[0][3] < y_mid + (0.1 * img.shape[0]):
                    segments.append(l)
                    if l[0][0] < x_mid and abs(x_mid - l[0][0]) < abs(x_mid - ltick[0][0]):
                        ltick = l
                    if l[0][0] > x_mid and abs(x_mid - l[0][0]) < abs(x_mid - rtick[0][0]):
                        rtick = l

        return (ltick, rtick, segments)

    def get_boxes_east(self, img):
        timg = img.copy()
        origH = timg.shape[0]/2
        if len(timg.shape) == 2:
            timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2RGB)
        (H, W) = timg.shape[:2]
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        timg = cv2.resize(timg, (newW, newH))
        (H, W) = timg.shape[:2]

        blob = cv2.dnn.blobFromImage(timg, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False) #dont touch magic nums
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)
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
                
                if endY*rH < origH:
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scores_data[x])

        boxes = non_max_suppression(np.array(rects), probs=confidences)
        return (boxes, rW, rH)

if __name__ == "__main__":
    sae = SpectrometerAngleEstimator()
    if len(sys.argv) > 1:
        print(sys.argv[1])
        sae.from_frame(cv2.imread(sys.argv[1]))
    else:
        sae.from_frame(cv2.imread("images/test2.jpg"))
