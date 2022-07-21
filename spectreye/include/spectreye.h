// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#pragma once

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>


/*
 * Location of prebuilt EAST OCR model.
 * Don't change this default! If you want a differnt path, pass it to CMake.
 */
#ifndef EAST_PATH
	#define EAST_PATH "../data/east.pb"		
#endif

const double HMS_MIN  = 10.5;
const double HMS_MAX  = 90.0;
const double SHMS_MIN =  5.5;
const double SHMS_MAX = 35.0; 

/*
 *	If the difference between the encoder angle and OCR angle is more than this, the composite
 *	value will be returned. Operates under the assumption that the encoder generally is not off by
 *	a huge amount.
 */
const double MARK_THRESH = 1.0;


/*
 *	Possible status that a SpectreyeReading can have. Always check this before operating on 
 *	the returned angle. Because of C++11 compliance, std::optional cannot be used for angle 
 *	results, so it's important to verify the status of the returned angle.
 */
enum RetCode 
{
	RC_FAILURE = -1,	// No angle was determined.
	RC_SUCCESS =  0,	// Angle was found with OCR.
	RC_NOREAD  =  1,	// Angle was found, but encoder mark was used due to OCR read failure.
	RC_EXCEED  =  2 	// Angle was found, but difference exceeded MARK_THRESH, so returned enc.
};

/*
 *	Either SHMS or HMS. If a value of DT_UNKNOWN is passed to Spectreye it will assume HMS.
 *	You should always use FindAngleHMS or FindAngleSHMS so it shouldn't be problem.
 */
enum DeviceType 
{
	DT_UNKNOWN = -1,	// Uknown device type. This should never happen.
	DT_HMS     =  0,	// SHMS image. Ticks are read right-to-left.
	DT_SHMS    =  1		// HMS image. Ticks are read left-to-right.
};

/*
 * 	Result data from FromFrame. Don't just use `angle`, make sure to verify status first.
 * 	If you suspect that Spectreye selected the wrong guess, you can access both.
 */
struct SpectreyeReading {
	RetCode 	status;		// Result of attempted read. Should always be checked.	
	DeviceType  dev_type;	// SHMS or HMS. Ideally DT_UNKNOWN is never used.
	std::string filename;	// Absolute path of image.
	std::string timestamp;	// Snapshot timestamp extracted from image file to use with enocder csv.

	double angle;		// Final angle guess, either OCR or composite.

	double ocr_guess;	// Angle calculated using OCR-read angle mark.
	double comp_guess;	// Angle calculated using angle mark nearest to encoder value.
	
	double mark;		// OCR-read angle mark.
	double tick;		// Distance in degrees between center tick and angle mark.
};

/*
 * 	Class for use through the shared library. 
 * 	This is a class instead of just a method so that models and other data can be loaded into
 * 	memory once and then accessed for multiple images.
 *
 *	An example of how to use the class can be found in the readme.
 *
 * 	Method descriptions can be found in ../src/spectreye.cpp
 */
class Spectreye 
{
private:
	bool debug = false;		// Debug mode will display visual result.
											
	int font = cv::FONT_HERSHEY_SIMPLEX;	// Font for numbers on visual result.
											
	int npadx = 50;		// Horizontal padding for OCR bounding boxes.
	int npady = 25;		// Vertical padding for OCR bounding boxes.

	std::vector<std::string> layer_names = {	// Layers for EAST optimized for text detection.
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"
	};

	cv::Mat okernel, dkernel, ckernel;		// Morphology elements for image filtering.
	cv::dnn::Net net;						// Neural net base class to load EAST into.
	cv::Ptr<cv::LineSegmentDetector> lsd;	// Model for detecting lines to begin mid detection.
	cv::Ptr<cv::CLAHE> clahe;				// Tool for boosting image contrast during filtering.

	/*
	 *	Tesseract OCR access point. Should be initialized once, and then tweaked based on intended
	 *	use. Different settings work better for bounding box detection vs reading numbers.
	 *	You can probably get more accuracy out of it with better settings than what are currently
	 *	being used.
	 */
	tesseract::TessBaseAPI *tess;	

	cv::Mat MaskFilter(cv::Mat frame);
	cv::Mat ThreshFilter(cv::Mat frame);
	cv::Mat CLAHEFilter(cv::Mat frame, int passes=1);
	std::vector<cv::Rect> OcrEast(cv::Mat frame);
	std::vector<cv::Rect> OcrTess(cv::Mat frame);
	int FindTickCenter(cv::Mat img, int ytest, int xtest, int delta=0);
	SpectreyeReading FromFrame(cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle);

public:
	Spectreye(int debug=false);
	SpectreyeReading GetAngleHMS(std::string path, double encoder_angle=0.0);
	SpectreyeReading GetAngleSHMS(std::string path, double encoder_angle=0.0);
	void Destroy();
	std::string ExtractTimestamp(std::string path);
	static std::string DescribeReading(SpectreyeReading r);

};
