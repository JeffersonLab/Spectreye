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

std::vector<cv::Rect> Spectreye::OcrEast(cv::Mat img) 
{
	std::vector<cv::Mat> outs;
	
	cv::Mat timg = img.clone();
	int H = timg.size().height;
	int W = timg.size().width;

	int origH = H;
	int newW, newH = 320;
	double rW = (double) W / (double) newW;
	double rH = (double) H / (double) newH;
	
	cv::resize(timg, timg, cv::Size(newW, newH));
	H = timg.size().height;
	W = timg.size().width;

	cv::Mat blob = cv::dnn::blobFromImage(
			timg, 1.0, cv::Size(H, W), cv::Scalar(123.68, 116.78, 103.94), true, false); 

	this->net.setInput(blob);
	this->net.forward(outs, this->layer_names);

	cv::Mat scores   = outs[0];
	cv::Mat geometry = outs[1];

	std::vector<cv::Rect> rects;
	std::vector<float> confidences;

	int nrows = scores.size[2];
	int ncols = scores.size[3];

	for(int y=0; y<nrows; ++y) {
		const float* scores_data = scores.ptr<float>(0, 0, y);
		const float* xdata0 = geometry.ptr<float>(0, 0, y);
		const float* xdata1 = geometry.ptr<float>(0, 1, y);
		const float* xdata2 = geometry.ptr<float>(0, 2, y);
		const float* xdata3 = geometry.ptr<float>(0, 3, y);
		const float* angles = geometry.ptr<float>(0, 4, y);

		for(int x=0; x<ncols; ++x) {
			if (scores_data[x] < 0.5) {
				continue;
			}
			float offsetX = x * 4.0;
			float offsetY = y * 4.0;
			float angle = angles[x];
			float cos = std::cos(angle);
			float sin = std::sin(angle);
			float h = xdata0[x] + xdata2[x];
			float w = xdata1[x] + xdata3[x];

			int endX = (int)(offsetX + (cos * xdata1[x]) + (sin * xdata2[x]));
			int endY = (int)(offsetY - (sin * xdata1[x]) + (cos * xdata2[x]));
			int startX = (int)(endX - w);
			int startY = (int)(endY - h);
			
			if (endY*rH < origH) {
				rects.push_back(cv::Rect(startX*rW, startY*rH, endX*rW, endY*rH));
				confidences.push_back(scores_data[x]);
			}
		}
	}

	std::vector<int> indices;
	cv::dnn::NMSBoxes(rects, confidences, 0.5, 0.5, indices);

	return rects;
}
/*
std::vector<cv::Rect> Spectreye::OcrTess(cv::Mat img) 
{
	
}

int Spectreye::FindTickCenter(cv::Mat img, int ytest, int xtest, int delta) 
{

}
*/
std::string Spectreye::FromFrame(
		cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle)
{
	return "";	
}

int main(int argc, char** argv) {
	Spectreye* s = new Spectreye(true);
	return 0;
}

