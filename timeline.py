import sys
import os
import subprocess as sp
import shutil
import ciso8601
import time

# helper script to create timeline csv with each angle change for data comparison

encodings = "datasets/HallC_SHMS_HMS_2018/HallC_SpectrometerAngles2018.dat"
sdest     = "datasets/angle_marks/shms/timeline.csv"
hdest     = "datasets/angle_marks/hms/timeline.csv"

def build_timeline():
    lines = open(encodings, "r").read().splitlines()[1:]
    h  = [] # hms angle snapshots
    s = []  # shms angle snapshots

    for i in range(0, len(lines)):
        words = lines[i].split()
        hms = words[2]
        shms = words[3]
        stamp = ciso8601.parse_datetime(words[0] + " " + words[1])
        ts = int(time.mktime(stamp.timetuple())) # unix timestamp from readable time in data

        if i == 0:
            h.append([ts, hms])
            s.append([ts, shms])
            continue

        if h[-1][1] != str(hms):
            h.append([ts, hms])
        if s[-1][1] != str(shms):
            s.append([ts, shms])

    sfile = open(sdest, "w+")
    hfile = open(hdest, "w+")

    for l in h:
        hfile.write(str(l[0]) + "," + str(l[1]) + "\n")
    for l in s:
        sfile.write(str(l[0]) + "," + str(l[1]) + "\n")

    sfile.close()
    hfile.close()

    
if __name__ == "__main__":
    build_timeline()
