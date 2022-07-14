import numpy as np
import cv2
import math
import sys
import os
import time
import json
import subprocess as sp
from enum import Enum
from imutils.object_detection import non_max_suppression
import pytesseract
from pytesseract import Output

# -- Spectreye --
# Spectreye is a tool for automatically determining the angle of the Super High Momentum Spectrometer 
# (SHMS) and the High Momentum Spectrometer (HMS) in JLab's Experimental Hall C. This script uses
# OpenCV and Pytesseract to extract precise angle readings from images taken by cameras mounted on
# the spectrometers' Vernier calipers. The script is optimized for the JLab spectrometers, but can 
# most likely be adapted to read images from any Vernier scale as long as the center of the caliper
# is near the center of the image.
# ---------------


# return codes for json object
class RetCode(Enum):
    FAILURE = -1 # could not read at all
    SUCCESS =  0 # successfully found both angle and tick
    NOREAD  =  1 # couldn't read angle mark but calculated tick distance

# this should ideally always be known, but some training images are unknown
class DeviceType(Enum):
    UNKNOWN = -1
    HMS     =  0
    SHMS    =  1

class Spectreye(object):

    # angle ranges. consider failure if outside bounds, or move decimal
    HMS_MIN = 10.5
    HMS_MAX = 90.0
    SHMS_MIN = 5.5
    SHMS_MAX =  35

    # EAST dnn layers
    layer_names = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"]

    npad = 100 
    font = cv2.FONT_HERSHEY_SIMPLEX
    stamps = []
    
    # loads dnn model into memory and builds morphology elements
    def __init__(self, debug=True):
        self.stamp("init")
        self.dkernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (4,4))
        self.okernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
        self.ckernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,4))

        self.net = cv2.dnn.readNet("east.pb")
        self.lsd = cv2.createLineSegmentDetector(0)
        self.debug = debug

    # adds messaged timestamp for runtime analysis
    def stamp(self, msg):
        self.stamps.append([msg, time.time()])

    # prints runtime of each stamp in delta seconds from last
    # TODO real logging system
    def disp_rt(self):
        self.stamp("final")
        print("\nruntime stamps\n--------------")
        for i in range(1, len(self.stamps)):
            delta = round(self.stamps[i][1] - self.stamps[i-1][1], 5)
            print(self.stamps[i][0] + " - delta: " + str(delta))
        total = round(self.stamps[len(self.stamps)-1][1] - self.stamps[0][1], 5)
        print("runtime: " + str(total))
        print("--------------\n")

    # wrapper for reading from image
    def from_image(self, path, cmp=None, dtype=DeviceType.UNKNOWN):
        if dtype == DeviceType.UNKNOWN:
            if "SHMS" in path:
                dtype = DeviceType.SHMS
            elif "HMS" in path:
                dtype = DeviceType.HMS

        return self.from_frame(cv2.imread(path), dtype=dtype, ipath=path)

    # extracts angle from single frame. can be used for both images and video
    def from_frame(self, frame, dtype=DeviceType.UNKNOWN, ipath=None):
        self.stamps = []
        self.stamp("from_frame begin")
        x_mid = frame.shape[1]/2
        y_mid = frame.shape[0]/2

        if ipath != None:
            if "_SHMS" in ipath:
                dtype=DeviceType.SHMS
            elif "_HMS" in ipath:
                dtype=DeviceType.HMS

        timestamp = extract_timestamp(ipath) if ipath != None else None

        img = frame
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ltick and rtick determine the bounds of the first suspected middle
        # tick, and are used as the basis for peak iteration
        self.stamp("get_ticks begin")
        (ltick, rtick, segments) = self.get_ticks(img, False)
        self.stamp("get_ticks end")

        # determines vernier '0' location, as well as pixel/degree ratio
        (true_mid, pixel_ratio, ysplit) = self.find_mid(img, segments, ltick, rtick)
        self.stamp("find_mid end") 

        #print("0.01 degrees = " + str(pixel_ratio) + " pixels")
        
        final = self.lsd.drawSegments(img, np.asarray([segments]))
        cv2.rectangle((final), (int(true_mid), 0), (int(true_mid), img.shape[0]), (255, 255, 0), 1) 

        # main filter pass for both EAST and tesseract. feel free to experiment,
        # but if you change it, keep the dilation and division together
        
#        cv2.imshow("t", timg)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()

#        self.ocr_tess(frame.copy())

        pass1 = self.thresh_filter(img)
        timg = pass1

        (boxes, rW, rH) = self.ocr_east(timg)
        self.stamp("ocr_east end")

        if len(boxes) == 0:
            (boxes, rW, rH) = self.ocr_tess(timg)

        if len(boxes) == 0:
            timg = self.mask_filter(frame, False)
            (boxes, rW, rH) = self.ocr_east(timg)
            if len(boxes) == 0:
                (boxes, rW, rH) = self.ocr_tess(timg)

        if len(boxes) == 0:
            if self.debug:
                rawnum = pytesseract.image_to_string(timg, lang="eng", config="--psm 6")
                print(rawnum)
                cv2.imshow("Failure", timg)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print("Could not locate any numbers!")
            return self.build_res(dtype=dtype, ts=timestamp)

        # creates boxes around east guesses
        self.stamp("draw boxes begin")
        cmpX = 0
        botY = 0
        bH = 0
        boxdata = None
        for(startX, startY, endX, endY) in boxes:
            startX = int(startX * rW)
            startY = int(startY * rH)
            endX = int(endX * rW)
            endY = int(endY * rH)
            cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # center of text bounding box
            tpos = startX + (endX-startX)/2

            # find the bounding box nearest to center to use as our number
            if abs(x_mid - tpos) < abs(x_mid - cmpX):
                cmpX = tpos
                botY = endY
 
                width = abs(endX-startX)
                height = abs(endY-startY)
                bH = height/2

                # extra padding so numbers dont get cut off
                boxdata = [
                        max(0, startY-50), 
                        min(frame.shape[0], endY+self.npad), 
                        max(0, startX-self.npad), 
                        min(frame.shape[1], endX+self.npad)
                ]

        cv2.rectangle(final, (startX, startY), (endX, endY), (0, 255, 0), 2)
        self.stamp("draw boxes end")

        # finds nearest "big tick" to selected text box using y midpoint 
        tseg = segments[0]
        for l in segments:
            if abs(cmpX - l[0][0]) < abs(cmpX - tseg[0][0]) and l[0][1] > botY and l[0][1] < ysplit:
                tseg = l
        tmidy = int(tseg[0][1] + ((tseg[0][3]-tseg[0][1])/4)*3)

        tick = tseg[0][0]

        if tick > endX or tick < startX:
            tick = startX + width/2
#            tick = self.proc_peak(pass1, tmidy)

        cv2.rectangle((final), (int(tick), 0), (int(tick), frame.shape[0]), (255, 0, 0), 1)   
        #self.proc_peak(pass1, tmidy) 
        
        pix_frac = true_mid - tick
        dec_frac = (pix_frac/pixel_ratio)*0.01
       
#        print("Additional distance (deg): " + str(dec_frac))
       
        # isolate text box for additional filtering to improve image for tesseract
        numbox = pass1[boxdata[0]:boxdata[1], boxdata[2]:boxdata[3]]

        #numbox = cv2.fastNlMeansDenoising(numbox,None,21,7,21)
       

        img = numbox
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        numbox = img

#        numbox = cv2.GaussianBlur(numbox, (3, 3), 5)
#        nmbox = cv2.fastNlMeansDenoising(numbox,None,21,7,21)
#        numbox = cv2.morphologyEx(numbox, cv2.MORPH_OPEN, self.okernel)
#        numbox = cv2.threshold(numbox, 240, 255, cv2.THRESH_BINARY, 0)[1] 
#        numbox = cv2.fastNlMeansDenoising(numbox,None,21,7,21)

        self.stamp("tess begin")
        rawnum = pytesseract.image_to_string(numbox, lang="eng", config="--psm 6")
        self.stamp("tess end")

        nstr = ""
        #print(rawnum)
        for n in rawnum:
            if n.isdigit():
                nstr +=n
            if n == '\n':
                break
        if len(nstr) < 3:
            nstr = nstr + ".0"
        else:
            nstr = nstr[:2] + "." + nstr[2:]

        tickR = 1 #SHMS ticks are inverse of HMS
        mark = float(nstr)
        if mark < self.SHMS_MIN:
            angle = None
        else:
            if dtype == DeviceType.SHMS:
                tickR = -1
                if mark > self.SHMS_MAX:
                    mark = mark / 10
                if mark < self.SHMS_MIN:
                    mark = None
            elif dtype == DeviceType.HMS:
                if mark > self.HMS_MAX:
                    mark = mark / 10
            
            angle = round(mark + (tickR * dec_frac), 2)

        #print(angle) 

        if self.debug:
            self.disp_rt()
            print("calculated angle: " + str(angle) + " degrees")
            cv2.putText(final, str(angle), (10, 30), self.font, 1, (0, 255, 0), 2, 2)

            #display and poll
            cv2.imshow("Detector", final)
        
            cv2.imshow("numbox", numbox)

            while True:
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            cv2.destroyAllWindows()

        fpath = os.path.abspath(ipath) if ipath != None else ipath
        obj = self.build_res(angle=angle, name=fpath, tick_angle=dec_frac, reading=float(nstr), dtype=dtype, ts=timestamp)
        if self.debug:
            print(obj)
        return obj

    # create json return object with angle data and success result
    def build_res(self, name=None, angle=None, tick_angle=None, reading=None, dtype=DeviceType.UNKNOWN, ts=None):
        if angle != None:
            status = RetCode.SUCCESS
            angle = str(round(angle, 2))
        elif tick_angle != None:
            status = RetCode.NOREAD
        else:
            status = RetCode.FAILURE

        if reading != None:
            reading = str(round(reading, 2))
        if tick_angle != None:
            tick_angle = str(round(tick_angle, 2))

        dat = {
            'status': status.name,
            'name': name,
            'angle': angle,
            'mark': reading,
            'tick': tick_angle,
            'runtime': str(round(self.stamps[len(self.stamps)-1][1] - self.stamps[0][1], 4)),
            'device': dtype.name,
            'timestamp': ts
        }
        return json.dumps(dat, indent=4)

    # test filtering based on color mask - messier numbers but consistent filter accuracy
    def mask_filter(self, frame, debug=False):
        fil = frame.copy()
        lab = cv2.cvtColor(fil, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        fil = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        fil = cv2.fastNlMeansDenoising(fil,None,21,7,21)

        mask = cv2.inRange(fil, (0, 0, 0), (190, 190, 250))
        fil = cv2.bitwise_or(fil,fil, mask=mask)
        fil = cv2.cvtColor(fil, cv2.COLOR_BGR2GRAY)
        fil = cv2.threshold(fil, 1, 255, cv2.THRESH_BINARY, 0)[1]

        fil = cv2.morphologyEx(fil, cv2.MORPH_OPEN, self.okernel)
        fil = cv2.GaussianBlur(fil, (5, 5), 5)

        if debug:
            cv2.imshow("f", fil)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return fil

    def thresh_filter(self, pass1):
        self.stamp("main pass begin") 
        pass1 = cv2.threshold(pass1, 127, 255, cv2.THRESH_BINARY_INV, 0)[1]
        bg = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, self.dkernel)
        pass1 = cv2.divide(pass1, bg, scale=255)
        pass1 = cv2.GaussianBlur(pass1, (3, 3), 0)
        pass1 = cv2.morphologyEx(pass1, cv2.MORPH_OPEN, self.okernel)
        pass1 = cv2.GaussianBlur(pass1, (5, 5), 0)
        pass1 = cv2.fastNlMeansDenoising(pass1,None,21,7,21)
        self.stamp("main pass end")
        return pass1

    # finds starting point for proc_peak based on l/r tick guesses
    def find_mid(self, img, segments, ltick, rtick):
        pass1 = img

        pass1 = cv2.GaussianBlur(pass1, (5, 5), 0)
        pass1 = cv2.fastNlMeansDenoising(pass1,None,21,7,21)
#        cv2.imshow("t", pass1)
#        cv2.waitKey(0)

        mid = segments[0]
        x_mid = img.shape[1]/2 

        # finding mid has multiple stages. first stage tries the tallest segment
        # nearby ltick/rtick. this works on optimally set up images, but fails
        # when the camera is poorly aligned
        opts = []
        for l in segments:
            if ltick[0][1] >= l[0][1] and ltick[0][3] <= l[0][3]:
                opts.append(l)
                if l[0][3] - l[0][1] > mid[0][3] - mid[0][1]:
                    mid = l

        # gets rid of outliers on y axis
        cull = []
        for l in opts:
            if not (l[0][1] < ltick[0][1]-(ltick[0][3]-ltick[0][1])):
                cull.append(l)
            elif not (l[0][3] > ltick[0][3]+(ltick[0][3]-ltick[0][1])):
                cull.append(l)
        
        # finds closest segment to image mid (not true mid)
        for l in cull:
            if abs(x_mid - l[0][0]) < abs(x_mid - mid[0][0]):
                mid = l
     
        # gets peak-based pixel width and mid prediction using original mid
        # position as starting point for peak trawling 
        width, peak_mid, ysplit = self.proc_peak(pass1, int(mid[0][1] + (mid[0][3]-mid[0][1])/2))
 
        return (peak_mid, width, ysplit)

    # locates probable midpoint and tick width at a given y
    def proc_peak(self, img, y):
        x_mid = int(img.shape[1]/2)

        # gets all peaks where alpha change > 2 and y = y
        ticks = []
        for x in range(0, img.shape[1]-1):
            if img[y][x] > img[y][x+1]+1 and img[y][x] > img[y][x-1]+1:
                ticks.append(x)

        # numpy magic to find the most common difference between peaks, which we
        # take as the tick width in pixels
        diffs = np.diff(np.asarray(ticks))
        width = np.bincount(diffs).argmax()

        # for each peak, finds the height of corresponding line by walking the image
        # up and down until it suddenly gets significantly darker
        heights = []
        ysplit = 0
        for l in ticks:
            uy = y
            while img[uy][l] < img[uy-1][l]+5:
                uy -= 1
            dy = y
            ysplit = uy
            while img[dy][l] < img[dy+1][l]+5:
                dy += 1
            heights.append(dy-uy)

        # funky numpy stuff that sorts by height and takes the indicies of the 5 tallest
        # ticks as an index slice that can be used to cull each array 
        heights = np.asarray(heights)
        opti = list(np.argpartition(heights, -5)[-5:] )
        ticks = np.asarray(ticks)
        locs = np.asarray(ticks[opti])
        heights = np.asarray(heights[opti])
        
        # finds distance from image mid of each tick
        dists = []
        for l in locs:
            dists.append(abs(x_mid - l))
        dists = sorted(dists, reverse=True)

        # i think i can just zip these values but i haven't tried
        fin = []
        for i in range(0, len(heights)):
            fin.append([locs[i], heights[i], dists[i]])

        # takes 3 ticks closest to image mid and then sorts by height
        res = sorted(fin, key=lambda x: abs(x_mid - x[0]))[0:2]
        cull = sorted(res, key=lambda x: x[0])

        #tallest of 3 final options is the best guess for the true middle of the image
        pred_mid = cull[0][0]

        return (abs(width), pred_mid, ysplit)

    # uses built-in line segment detector to roughly locate hundredths ticks
    def get_ticks(self, img, debug=False):
        x_mid = img.shape[1]/2
        y_mid = img.shape[0]/2

        pass1 = img
        pass1 = cv2.fastNlMeansDenoising(pass1,None,21,7,21)
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

            # removes all horizontally biased lines and only keeps lines in middle 20% of image
            if abs(l[0][0] - l[0][2]) < abs(l[0][1] - l[0][3]):
                if l[0][1] > y_mid - (0.1 * img.shape[0]) and l[0][3] < y_mid + (0.1 * img.shape[0]):
                    segments.append(l)

                    # ltick and rtick are the two segments closest to the image center on either side
                    if l[0][0] < x_mid and abs(x_mid - l[0][0]) < abs(x_mid - ltick[0][0]):
                        ltick = l
                    if l[0][0] > x_mid and abs(x_mid - l[0][0]) < abs(x_mid - rtick[0][0]):
                        rtick = l

        return (ltick, rtick, segments)

    # looks for numbers bounding boxes using the EAST dnn
    def ocr_east(self, img):
        timg = img.copy()
        origH = timg.shape[0]/2

        # east expects all incoming images to be a square multiple of 32
        if len(timg.shape) == 2:
            timg = cv2.cvtColor(timg, cv2.COLOR_GRAY2RGB)
        (H, W) = timg.shape[:2]
        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        timg = cv2.resize(timg, (newW, newH))
        (H, W) = timg.shape[:2]

        # configures data blob from the image using predetermined magic nums
        blob = cv2.dnn.blobFromImage(timg, 1.0, (W, H),
                (123.68, 116.78, 103.94), swapRB=True, crop=False) 
        self.net.setInput(blob)
        (scores, geometry) = self.net.forward(self.layer_names)
        (nrows, ncols) = scores.shape[2:4]
        rects = []
        confidences = []

        # takes each returned OCR prediction from east and converts from polar
        # to cartesian coordinates (why does east uses polar coords?? i dont know)
        for y in range(0, nrows):
            scores_data = scores[0, 0, y]
            xdata0 = geometry[0, 0, y]
            xdata1 = geometry[0, 1, y]
            xdata2 = geometry[0, 2, y]
            xdata3 = geometry[0, 3, y]
            angles = geometry[0, 4, y]

            for x in range(0, ncols):
                
                # you can play with the acceptance threshold, but 0.5 seems to work best
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
                
                # some wacky predictions go off the image, get rid of them
                if endY*rH < origH:
                    rects.append((startX, startY, endX, endY))
                    confidences.append(scores_data[x])

        # combine nearby rects that are most likely the same text into one
        # https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c
        self.stamp("start NMS")
        boxes = non_max_suppression(np.array(rects), probs=confidences)
        self.stamp("end NMS")
        return (boxes, rW, rH)

    # looks for bounding boxes using pytesseract built-in ocr->rect
    def ocr_tess(self, img):
        print("using tess ocr")
        # correct lighting errors with CLAHE algorithm
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


        img = cv2.fastNlMeansDenoising(img,None,21,7,21) 
        img = cv2.GaussianBlur(img, (3, 3), 5)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.fastNlMeansDenoising(img,None,21,7,21)
        img = cv2.GaussianBlur(img, (3, 3), 5)
        img = cv2.fastNlMeansDenoising(img,None,21,7,21)
#        img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY, 0)[1] 

        self.stamp("begin ocr_tess")
        d = pytesseract.image_to_data(img, output_type=Output.DICT, lang="eng", config="--psm 6")
        n_boxes = len(d['level'])
        rects = []
        confs = []
        for i in range(n_boxes):
            if float(d['conf'][i]) > 40 and d['top'][i]+d['height'][i] <= img.shape[0]/2:
                if float(d['width'][i]) > float(d['height'][i]):
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    rects.append((x, y, x+w, y+h))
                    confs.append(float(d['conf'][i])/100)

        boxes = zip(rects, confs)
        #boxes = non_max_suppression(np.array(rects), probs=confs)
        #boxes = sorted(zip(rects, confs), key=lambda x: x[1], reverse=True)[0:int(len(confs)/2)]
        #print(boxes)
        #boxes = sorted(boxes, key=lambda x: x[0][2]*x[0][3], reverse=True)[0:1]
        #print(boxes)

        for ((x, y, ex, ey), c) in boxes:
            cv2.rectangle(img, (x, y), (ex, ey), (0, 255, 0), 2)

        #cv2.imshow("tess", img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        self.stamp("end ocr_tess")
        return (rects, 1, 1) # resize ratio of 1 for plug-n-play compatibility with ocr_east calls

    # finds the closest peaks on the x axis in both directions and returns the closest
    def find_tick_center(self, img, ytest, xtest, delta=0):
        optl, optr = 0, 0
        for x in reversed(range(1, xtest)):
            if img[ytest][x] > img[ytest][x-1]+delta:
                optl = x
                break
        for x in range(xtest, img.shape[1]-1):
            if img[ytest][x] > img[ytest][x+1]+delta:
                optr = x
                break
        midx = optl if (abs(xtest-optl) > abs(xtest-optr)) else optr

        return midx

# uses gnu strings to locate the first timestamp string from within a binary
# only tested on angle_snap images, but should work for most jpgs 
# MUST use -P flag for perl-style regex to work
def extract_timestamp(path):
    datetime = sp.getoutput("strings " + path + " | grep -P \"(19|20)[\d]{2,2}\"").splitlines()[0]
    datetime = datetime.replace(":", "-", 2)
    return datetime

if __name__ == "__main__":
    sae = Spectreye()
    if len(sys.argv) > 1:
        print(sys.argv[1])
        res = sae.from_image(sys.argv[1])
    else:
        sae.from_image("images/qtest/HMS_0.jpg", cmp=19.68)
        
        
