#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include "spectreye.h"


Spectreye::Spectreye(int debug) 
{
	this->dkernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(4,4));
	this->okernel = cv::getStructuringElement(cv::MORPH_OPEN, cv::Size(1,1));
	this->ckernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,4));

	this->net = cv::dnn::readNet("../east.pb");
	this->lsd = cv::createLineSegmentDetector(0);

	this->clahe = cv::createCLAHE();
	this->clahe->setClipLimit(2.0);
	this->clahe->setTilesGridSize(cv::Size(8, 8));

	this->debug = debug;
}

std::string Spectreye::GetAngleHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_HMS, path, encoder_angle);
}

std::string Spectreye::GetAngleSHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_SHMS, path, encoder_angle);
}

std::string Spectreye::ExtractTimestamp(std::string path) 
{
	return "not implemented";
}

cv::Mat Spectreye::ThreshFilter(cv::Mat frame) {
	cv::Mat img;
	cv::threshold(frame, img, 127, 255, cv::THRESH_BINARY_INV);
	cv::morphologyEx(img, img, cv::MORPH_DILATE, this->dkernel);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::morphologyEx(img, img, cv::MORPH_OPEN, this->okernel);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	return img;
}

cv::Mat Spectreye::MaskFilter(cv::Mat frame) {
	cv::Mat img, lab, mask;
	cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
	
	std::vector<cv::Mat> lplanes(3);
	cv::split(lab, lplanes);
	cv::Mat dst;
	this->clahe->apply(lplanes[0], dst);
	dst.copyTo(lplanes[0]);
	cv::merge(lplanes, lab);
	cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);

	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	cv::inRange(img, cv::Scalar(0, 0, 0), cv::Scalar(190, 190, 250), mask);
	cv::bitwise_or(img, img, img, mask=mask);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::threshold(img, img, 1, 255, cv::THRESH_BINARY);
	cv::morphologyEx(img, img, cv::MORPH_OPEN, this->okernel);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);

	return img;
}

std::string Spectreye::FromFrame(
		cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle)
{
	return "";	
}

int main(int argc, char** argv) {
	Spectreye* s = new Spectreye(true);
	return 0;
}

