import sys
import os
import random
import cv2
import subprocess
import json
from spectreye import Spectreye, DeviceType
from timeline import cmp_reading

# helper tests for running Spectreye on different image batches

ds = "datasets/HallC_SHMS_HMS_2018/HallC_SpectrometerAngles2018.dat"

lowqual = ["COIN_HMS_angle_03322.jpg", "COIN_HMS_angle_03814.jpg", "HMS_angle_01870.jpg"]

# choose randomly from preselected test images
def gtest(sae):
    images = os.listdir("images/")
    random.shuffle(images)
    for path in images:
        if len(path) > 4 and path[-4:] == ".jpg":
            path = "images/" + path
            print(path)
            sae.from_frame(cv2.imread(path))

# quick gtest checker
def qtest():
    sae = Spectreye(False)
    vals = [19.68, 20.96, 47.61, 33.02, 28.51, 19.71, 21.41]
    angles = []
    times = []
    for i in range(0, len(vals)):
        path = "images/qtest/" + os.listdir("images/qtest")[i]
        print(path)
        res = json.loads(sae.from_image(path))
        print(res)
        angle = float(res.get("angle"))
        angles.append(angle)
        t = res.get("runtime")
        times.append(t)

    print("vis  | spectreye")
    for i in range(0, len(angles)):
        print(str(vals[i]) + " " + str(angles[i]) + " " + str(times[i])[0:5] + "s")

# choose randomly from all angle snaps
def rtest(sae):
    while True:
        path = random.choice(os.listdir("images/angle_snaps/"))
        if len(path) > 4 and path[-4:] == ".jpg":
            path = "images/angle_snaps/" + path
            print(path)
            res = sae.from_image(path)
            reading = cmp_reading(res)


# choose randomly from selected problem images
def prob_test(sae):
    while True:
        path = "images/angle_snaps/" + random.choice(lowqual)
        print(path)
        sae.from_frame(cv2.imread(path))

# choose randomly from SHMS snaps
def shms_test(sae):
    while True:
        path = random.choice(os.listdir("images/angle_snaps/"))
        print(path)
        if len(path) > 4 and path[-4:] == ".jpg" and "_SHMS" in path:
            path = "images/angle_snaps/" + path
            if path == "":
                continue

            res = sae.from_image(path)
            reading = cmp_reading(res)

# choose randomly from HMS snaps
def hms_test(sae):
    while True:
        path = random.choice(os.listdir("images/angle_snaps/"))
        print(path)
        if len(path) > 4 and path[-4:] == ".jpg" and "_HMS" in path:
            path = "images/angle_snaps/" + path
            if path == "":
                continue

            res = sae.from_image(path)
            reading = cmp_reading(res)


if __name__ == "__main__":
    sae = Spectreye(False)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "g" or sys.argv[1] == "-g":
            gtest(sae)
        elif sys.argv[1] == "q" or sys.argv[1] == "-q":
            qtest()
        elif sys.argv[1] == "r" or sys.argv[1] == "-r":
            rtest(sae)
        elif sys.argv[1] == "p" or sys.argv[1] == "-p":
            prob_test(sae)
        elif sys.argv[1] == "s" or sys.argv[1] == "-s":
            shms_test(sae)
        elif sys.argv[1] == "h" or sys.argv[1] == "-h":
            hms_test(sae)
