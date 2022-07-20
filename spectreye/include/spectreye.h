// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#pragma once

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <optional>

const double HMS_MIN  = 10.5;
const double HMS_MAX  = 90.0;
const double SHMS_MIN =  5.5;
const double SHMS_MAX = 35.0; //actual max is not actually 25 as documents purport. haven't seen >35

const double MARK_THRESH = 1.0;

enum RetCode 
{
	RC_FAILURE = -1,
	RC_SUCCESS =  0,
	RC_NOREAD  =  1,
	RC_EXCEED  =  2 
};

enum DeviceType 
{
	DT_UNKNOWN = -1,
	DT_HMS     =  0,
	DT_SHMS    =  1
};

struct SpectreyeReading {
	RetCode 	status;
	DeviceType  dev_type;
	std::string filename;
	std::string timestamp;

	double angle;

	double ocr_guess;
	double comp_guess;
	
	double mark;
	double tick;
};


class Spectreye 
{
private:
	bool debug = false;
	int font = cv::FONT_HERSHEY_SIMPLEX;
	int npadx = 50;
	int npady = 25;

	std::vector<std::string> layer_names = {
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"
	};

	cv::Mat okernel, dkernel, ckernel;
	cv::dnn::Net net;
	cv::Ptr<cv::LineSegmentDetector> lsd;
	cv::Ptr<cv::CLAHE> clahe;

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
	std::string ExtractTimestamp(std::string path);
	static std::string DescribeReading(SpectreyeReading r);

};
