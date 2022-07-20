# Spectreye
<img src="images/testgraphic.png" />

## Intro
Spectreye is a tool for automatically determining the angle of the Super High Momentum Spectrometer (SHMS) and the High Momentum Spectrometer (HMS) in JLab's Experimental Hall C. The program uses computer vision and optical character recognition to determine the angle of the spectrometers from photos of their Vernier calipers. A non-technical overview of the project can be found [here](https://docs.google.com/presentation/d/1qKy9npTbnCOFVQCxMHfdYh_vlz-lOZ7rnzxkA6Q_Qw8/edit#slide=id.gf489b73e5b_1_0).

This repository contains the C++ source of the shared library and CLI tool, the Python source of the prototype, a variety of angle images for testing, and the corresponding SHMS/HMS encoder data. Many of the Python tests that incorporate the encoder data have not yet been ported to C++.

Spectreye is functional, and can be integrated and utilized in existing codebases. The program works best when presented with relatively clean angle images and their corresponding encoder values. The OCR solutions used by Spectreye are not yet reliable enough for the program's output to be completely trusted, especially if it is not provided with an encoder reading. In its current state, Spectreye would be most useful if used to detect possible encoder drift and slippage, rather than as a primary source of spectrometer angle data.

Spectreye is licensed under the standard Jefferson Lab software license.

## Building
### Dependencies 
Spectreye requires: 
 - `OpenCV    >= 4.5.0`
 - `Tesseract >= 5.2.0`
 - `CMake     >= 2.8.0`

It most likely is not necessary to build any dependencies from source. The latest version in your system package manager should be fine.

```bash
git clone https://github.com/ws-kj/Spectreye.git
cd Spectreye
./setup.sh
```
### Building the shared library
CMake will attempt to install `libspectreye.so` in `/usr/lib`. This may be undesirable on shared systems, and can be changed in the `CMakeLists.txt`.

Spectreye must know the location of the file `spectreye/data/east.pb` to run. 
If you plan to move or delete the `Spectreye/` directory, you must store `east.pb` somewhere safe, and pass its location to CMake with `-DEAST_PATH="path/to/east"`.
```bash
# To build shared library (libspectreye.so)
cd spectreye
cmake .
make install
```
### Building the executable
The executable `spectreye-cli` requires the library to be installed. 
```bash
cd spectreye-cli
cmake .
make
```

## Using Spectreye
### Using the library
```C++
#include <iostream>
#include <spectreye.h>

...
std::string myimage = "SHMS_image.jpg";
double encoder_val = 45.23;

// argument specifies debug mode, true will display image
Spectreye* s = new Spectreye(true); 

// struct containing status, guesses, and other information
SpectreyeReading reading = s->GetAngleSHMS(myimage, encoder_val);

// formatted description 
std::cout << Spectreye::DescribeReading(reading) << std::endl;

if(reading.result != RC_FAILURE) {
	// do something with angle reading
	std::cout << reading.angle << std::endl;
}

// free allocations
s->Destroy();
delete s;
...
```

### Using the executable
```bash
# encoder value is optional but recommended
./spectreye-cli path/to/image.jpg [encoder value]

# -d flag will display image result
./spectreye-cli path/to/image [encoder value] -d
```
## Technical Overview
### Quick Enum Reference
 ```C++
// No angle was determined.
RC_FAILURE		
// Angle was found with OCR.
RC_SUCCESS		
// Angle was found, but encoder mark was used due to OCR read failure.
RC_NOREAD		
// Angle was found, but encoder mark was used due to excessive
// difference between encoder mark and OCR reading.
RC_EXCEED		

// Uknown device type. This should never happen.
DT_UNKNOWN
// SHMS image. Ticks are read right-to-left.
DT_SHMS
// HMS image. Ticks are read left-to-right.
DT_HMS
```

### Algorithm process
The general idea behind the algorithm is as follows:
 1. Locate 0 degree tick mark
 2. Determine 0.01 degree/pixel ratio
 3. Filter image for OCR
 4. Locate bounding box with OCR (Try both EAST and Tesseract before giving up)
 5. Attempt Tesseract read of angle mark
 6. Convert string returned by Tesseract into a valid angle
 7. Determine whether to return OCR or composite angle guess

If you want to understand the code,  `spectreye/src/spectreye.cpp` is generally well-commented, and should provide a good explanation for the decisions made during each step of the process.

## Further Improvements
While Spectreye is complete and usable, a number of improvements still need to be made to improve performance and reliability.
### Short term 
 - [ ] Change timestamp grep syntax to work without -P flag for MacOS/BSD
 - [ ] Reduce timestamp extraction overhead by loading images more efficiently
 - [ ] Improve error handling by strict checking OpenCV calls
 - [ ] Port encoder tests to C++
### Long term
 - [ ] Improve OCR results with more intelligent and dynamic filtering
 - [ ] Explore alternative OCR solutions, such as a model trained only on encoder angle marks
 - [ ] Optimize filtering by reducing unnecessary calls
 - [ ] Optimize tick recognition process with more efficient data structures
 - [ ] Bonus - optimize the program enough to the point that running Spectreye on video becomes viable

## 
Spectreye was written by Will Savage during the 2022 JLAB SRGS Internship program. 
Thank you to mentors Dr. David Lawrence, Mr. Nathan Brei, and Dr. Brad Sawatzky.

