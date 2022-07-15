import sys
import os
import subprocess as sp
import shutil
import ciso8601
import time
import json
from spectreye import DeviceType, RetCode, Spectreye
# helper script to create timeline csv with each angle change for data comparison

encodings = "../../datasets/HallC_SHMS_HMS_2018/HallC_SpectrometerAngles2018.dat"
stl = "../../datasets/angle_marks/shms/timeline.csv"
htl = "../../datasets/angle_marks/hms/timeline.csv"

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

    sfile = open(stl, "w+")
    hfile = open(htl, "w+")

    for l in h:
        hfile.write(str(l[0]) + "," + str(l[1]) + "\n")
    for l in s:
        sfile.write(str(l[0]) + "," + str(l[1]) + "\n")

    sfile.close()
    hfile.close()

# unpack json output from spectreye and compare angle, reading, and tick for error%
def cmp_reading(obj):
    data = json.loads(obj)

    if data.get("status") == RetCode.FAILURE.name:
        return

    dev = data.get("device")
    if dev == DeviceType.UNKNOWN.name:
        print("Unknown DeviceType")
        return
    elif dev == DeviceType.SHMS.name:
        tlpath = stl
        tR = -1
    else:
        tlpath = htl
        tR = 1

    # convert to unixtime
    words = data.get('timestamp').split()
    stamp = ciso8601.parse_datetime(words[0] + " " + words[1])
    ts = int(time.mktime(stamp.timetuple()))

    # compare time against angle changes to find correct range
    enc_ang = -1
    timeline = open(tlpath, "r").read().splitlines()
    for i in range(0, len(timeline)-1):
        atime, acmp = tuple(timeline[i].split(",")[0:2])
        ntime = timeline[i+1].split(",")[0]

        if ts > int(atime) and ts < int(ntime):
            enc_ang = float(acmp)

    if enc_ang == -1:
        print("Image timestamp outside of encoder data range (" + data.get('timestamp') + ")")
        return

    enc_mark = round(enc_ang * 2) / 2
    enc_tick = round(enc_ang - enc_mark, 2)

    # estimated angle assuming that encoder drift in < 0.5 degrees (which it should be (-: )
    if data.get("tick") != None:
        composite = round(enc_mark + (tR * float(data.get("tick"))), 2)
    else:
        composite = 0.0

    rstr = data.get("timestamp") 
    rstr += " (" + dev + ")\n"
    rstr += "encoder angle: " + str(enc_ang) + " deg. " 
    rstr += "spectreye angle: " + str(data.get("angle")) + " deg.\n"
    rstr += "encoder mark: " + str(enc_mark) + " deg. "
    rstr += "spectreye mark: " + str(data.get("mark")) + " deg.\n"
    rstr += "encoder tick: " + str(enc_tick) + " deg. "
    rstr += "spectreye tick: " + str(data.get("tick")) + " deg.\n"
    rstr += "composite guess: " + str(composite) + " deg.\n\n"

    print(rstr)
    return rstr

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sae = Spectreye(True)
        res = sae.from_image(sys.argv[1])
        read = cmp_reading(res)
    else:
        build_timeline()

