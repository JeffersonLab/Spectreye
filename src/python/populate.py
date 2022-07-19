# Copyright 2022, Jefferson Science Associates, LLC.
# Subject to the terms in the LICENSE file found in the top-level directory.


import sys
import os
import subprocess as sp
import shutil
import ciso8601
import time

# helper script to create and populate directories with SHMS/HMS images that have encodings.

encodings = "datasets/HallC_SHMS_HMS_2018/HallC_SpectrometerAngles2018.dat"
source    = "images/angle_snaps/"
dest      = "datasets/angle_marks/"
data      = "data.csv"
tldest    = "datasets/timeline.csv"

# matches all dates
regex_fmt = "[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])"

def build_timeline():
    lines = open(encodings, "r").read().splitlines()[1:]
    hms_segs  = []
    shms_segs = []

    for l in lines:
        stamp = ciso8601.parse_datetime(l.split()[0:2])
        ts = time.mktime(stamp.timetuple())
        print(ts)
        return
    

def populate_images():
    prelim = os.listdir(source)
    nfiles = len(prelim)
    
    if not os.path.isdir(dest):
        os.mkdir(dest)
    if not os.path.isdir(dest + "shms"):
        os.mkdir(dest + "/shms")
    if not os.path.isdir(dest + "hms"):
        os.mkdir(dest + "/hms")
 
    hfile = open(dest + "hms/data.csv", "w+")
    sfile = open(dest + "shms/data.csv", "w+")

    print("Spectreye -- populating image data directories...")
    c = 0
    for name in prelim:
        c += 1
        dpath = dest
        spath = source + name
        if "SHMS" in name:
            dpath += "shms/"
            shms = True
        else:
            dpath += "hms/"
            shms = False

        ts = sp.getoutput("strings " + spath + " | grep \"201\"").splitlines()[0] 
        ts = ts.replace(":", "-", 2) #match format with encoding
    
   #     print(ts)

        enc_raw = sp.getoutput("cat " + encodings + " | grep \"" + ts + "\"")
        if enc_raw == '':
            continue

#        print(enc_raw)

        # this will break if encoding file format ever changes :D
        enc_angle = (enc_raw[-5:].strip()) if shms else (enc_raw[27:32].strip())

 #       print(enc_angle)

        nearest = round(float(enc_angle) * 2)/2
        
  #      print(nearest)

        if shms: 
            sfile.write(name + "," + enc_angle + "," + str(nearest)+"\n")
        else: 
            hfile.write(name + "," + enc_angle + "," + str(nearest)+"\n")
       
        print("[" + str(c) + "]/[" + str(nfiles) + "] - " + name, end="\r")
#        shutil.copy2(spath, dest)

    hfile.close()
    sfile.close()
    print("\ndone.")


if __name__ == "__main__":
    populate_images()
