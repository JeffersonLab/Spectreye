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
const double SHMS_MAX = 35.0;

enum RetCode 
{
	RC_FAILURE = -1,
	RC_SUCCESS =  0,
	RC_NOREAD  =  1
};

enum DeviceType 
{
	DT_UNKNOWN = -1,
	DT_HMS     =  0,
	DT_SHMS    =  1
};

class Spectreye 
{
private:
	bool debug = false;
	int font = cv::FONT_HERSHEY_SIMPLEX;
	int npad = 25;
	
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
	std::vector<cv::Rect>OcrEast(cv::Mat frame);
	std::vector<cv::Rect>OcrTess(cv::Mat frame);
	int FindTickCenter(cv::Mat img, int ytest, int xtest, int delta=0);
	std::string FromFrame(cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle);

public:
	Spectreye(int debug=false);
	std::string GetAngleHMS(std::string path, double encoder_angle=0.0);
	std::string GetAngleSHMS(std::string path, double encoder_angle=0.0);
	std::string ExtractTimestamp(std::string path);

};
