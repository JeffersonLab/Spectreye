// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.


#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <cstdio>
#include <stdlib.h>

#include "spectreye.h"


Spectreye::Spectreye(int debug) 
{
	this->dkernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(4,4));
	this->okernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,1));
	this->ckernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,4));

	this->net = cv::dnn::readNet("../east.pb");
	this->lsd = cv::createLineSegmentDetector(0);

	this->clahe = cv::createCLAHE();
	this->clahe->setClipLimit(2.0);
	this->clahe->setTilesGridSize(cv::Size(8, 8));

	this->tess = new tesseract::TessBaseAPI();
	this->tess->Init(NULL, "eng");
	this->tess->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);
	this->debug = debug;
}

SpectreyeReading Spectreye::GetAngleHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_HMS, path, encoder_angle);
}

SpectreyeReading Spectreye::GetAngleSHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_SHMS, path, encoder_angle);
}

std::string Spectreye::ExtractTimestamp(std::string path) 
{
	std::string com = "strings " + path + " | grep -P \"(19|20)[\\d]{2,2}\"";

	char buf[128];
	std::string res;

	FILE* pipe = popen(com.c_str(), "r");
	if(!pipe)
		return "failed to build timestamp";

	while(!feof(pipe)) {
		if(fgets(buf, 128, pipe) != NULL)
			res += buf;
	}
	pclose(pipe);

	std::string datetime;
	int n = 0;
	for(const auto& c : res) {
		if(c != '\n')
			if(c == ':' && n < 2) {
				datetime += '-';
				n++;
			} else {
				datetime += c;
			}
		else
			break;
	}

	return datetime;
}

std::string Spectreye::DescribeReading(SpectreyeReading r) 
{
	std::stringstream ret;

	ret << "\n   Spectreye reading for \033[1;33m" << r.filename << "\033[1;0m\n\n";
	ret << "   ";
	switch(r.dev_type) {
		case DT_SHMS:
			ret << "\e[1mSHMS\e[0m";
			break;
		case DT_HMS:
			ret << "\e[1mHMS\e[0m";
			break;
		default:
			ret << "\e[1mUNKNOWN\e[0m";
	}
	ret << " - ";
	switch(r.status) {
		case RC_SUCCESS:
			ret << "\033[1;32mSUCCESS\033[1;0m";
			break;
		case RC_NOREAD:
			ret << "\033[1;33mNOREAD\033[1;0m";
			break;
		case RC_EXCEED:
			ret << "\033[1;33mEXCEED\033[1;0m";
			break;
		default:
			ret << "\033[1;31mFAILURE\033[1;0m";
			break;
	}
	ret << " - \033[1;34m";
	ret << std::fixed << std::setprecision(2) << r.angle;
	ret << " deg\033[1;0m\n";
	ret << "   --  Timestamp:  " << r.timestamp << std::endl;
	ret << std::fixed << ((r.ocr_guess != 0) ? std::setprecision(2) : std::setprecision(3));
	ret << "   --  OCR guess:  " << r.ocr_guess << " deg\n";
	ret << std::fixed << ((r.comp_guess != 0) ? std::setprecision(2) : std::setprecision(3));
	ret << "   --  Comp guess: " << r.comp_guess << " deg\n";
	ret << std::fixed << ((r.mark != 0) ? std::setprecision(2) : std::setprecision(3));
	ret << "   --  Angle mark: " << r.mark << " deg\n";
	ret << std::fixed << ((r.tick > 0) ? std::setprecision(3) : std::setprecision(2));
	ret << "   --  Tick count: " << r.tick << " deg\n";

	return ret.str();	
}

cv::Mat Spectreye::CLAHEFilter(cv::Mat frame, int passes) 
{
	cv::Mat img, lab;
	cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
	
	std::vector<cv::Mat> lplanes(3);
	cv::split(lab, lplanes);
	cv::Mat dst;
	for(int i=0; i<passes; i++) {
		this->clahe->apply(lplanes[0], dst);
		dst.copyTo(lplanes[0]);
	}
	cv::merge(lplanes, lab);
	cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);

	return img;
}

cv::Mat Spectreye::ThreshFilter(cv::Mat frame) 
{
	cv::Mat img, bg;
	cv::threshold(frame, img, 127, 255, cv::THRESH_BINARY_INV);
	cv::morphologyEx(img, bg, cv::MORPH_DILATE, this->dkernel);
	cv::divide(img, bg, img, 255);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::morphologyEx(img, img, cv::MORPH_OPEN, this->okernel);
	cv::GaussianBlur(img, img, cv::Size(5, 5), 0);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);

	return img;
}

cv::Mat Spectreye::MaskFilter(cv::Mat frame) 
{
	cv::Mat mask, img;
	img = this->CLAHEFilter(frame, 1);

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

std::vector<cv::Rect> Spectreye::OcrEast(cv::Mat frame) 
{
	std::vector<cv::Mat> outs;
	
	cv::Mat timg;
	cv::cvtColor(frame, timg, cv::COLOR_GRAY2BGR);

	int H = timg.size().height;
	int W = timg.size().width;

	int origH = H;
	int newW = 320, newH = 320;
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

	cv::Mat temp = frame.clone();

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
				cv::Rect rect = cv::Rect(startX*rW, startY*rH, (endX-startX)*rW, (endY-startY)*rH);

				cv::rectangle(temp, rect, cv::Scalar(0, 255, 0), 2);

				rects.push_back(rect);
				confidences.push_back(scores_data[x]);
			}
		}
	}
/*
	cv::imshow("R", temp);
	cv::waitKey(0);
	cv::destroyAllWindows();
*/
	std::vector<int> indices;
	cv::dnn::NMSBoxes(rects, confidences, 0, 0.5, indices);

	return rects;
}

std::vector<cv::Rect> Spectreye::OcrTess(cv::Mat frame) 
{
	std::cout << "Using Tess OCR" << std::endl;
	cv::Mat img, lab;	

	cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
	cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
	
	std::vector<cv::Mat> lplanes(3);
	cv::split(lab, lplanes);
	cv::Mat dst;
	this->clahe->apply(lplanes[0], dst);
	dst.copyTo(lplanes[0]);
	cv::merge(lplanes, lab);
	cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);

	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);

	this->tess->SetImage((unsigned char*)img.data, img.size().width, img.size().height, 
			img.channels(), img.step1());
	this->tess->Recognize(0);
	tesseract::ResultIterator* ri = this->tess->GetIterator();
	tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

	std::vector<cv::Rect> rects;

	if(ri != 0) {
		do {
			float conf = ri->Confidence(level);
			int x1, y1, x2, y2;
			ri->BoundingBox(level, &x1, &y1, &x2, &y2);

			if(conf > 40 && y2 <= img.size().height/2) {
				rects.push_back(cv::Rect(x1, y1, x2, y2));
			}

		} while (ri->Next(level));
	}

	return rects;
}

int Spectreye::FindTickCenter(cv::Mat img, int ytest, int xtest, int delta) 
{
	int optl = 0, optr = 0;

	for(int x=xtest-1; x>=1; x--) {
		if(img.at<unsigned char>(ytest, x) > img.at<unsigned char>(ytest, x-1)+delta) {
			optl = x;
			break;
		}
	}
	for(int x=xtest; x<img.size().width; x++) {
		if(img.at<unsigned char>(ytest, x) > img.at<unsigned char>(ytest, x+1)+delta) {
			optr = x;
			break;
		}
	}
	
	return (std::abs(xtest-optl) < std::abs(xtest-optr)) ? optl : optr;	
}

SpectreyeReading Spectreye::FromFrame(
		cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle)
{
	SpectreyeReading res;

	const char* icpy = ipath.c_str();
	const char* tpath = const_cast<char*>(icpy);
	ipath = std::string(realpath(tpath, NULL));

	std::string timestamp = this->ExtractTimestamp(ipath);

	res.dev_type = dtype;
	res.filename = ipath;
	res.timestamp = timestamp;

	int x_mid = frame.size().width/2;
	int y_mid = frame.size().height/2;

	cv::Mat img, display;
	cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);	
	

	cv::Vec4f ltick, rtick;
	std::vector<cv::Vec4f> segments;

	// equiv of get_ticks() from spectreye.py
	cv::Mat pass1;
	cv::fastNlMeansDenoising(img, pass1, 21, 7, 21);
	cv::GaussianBlur(pass1, pass1, cv::Size(5, 5), 0);
	
	std::vector<cv::Vec4f> lines;
	this->lsd->detect(pass1, lines);
	ltick = lines[0];
	rtick = lines[1];

	for(int i=0; i<lines.size(); i++) {
		cv::Vec4f l = lines[i];

		if(std::abs(l[0] - l[2]) < std::abs(l[1] - l[3])) {
			if(l[1] > y_mid - (0.1 * img.size().height) &&
					l[3] < y_mid + (0.1 * img.size().height)) {
				segments.push_back(l);

				if(l[0] < x_mid && std::abs(x_mid - l[0]) < std::abs(x_mid - ltick[0])) 
					ltick = l;	
				if(l[0] > x_mid && std::abs(x_mid - l[0]) < std::abs(x_mid - rtick[0])) 
					rtick = l;
			}
		}
	}

	int true_mid, ysplit;
	float pixel_ratio;

	// equiv of find_mid from spectreye.py	
	cv::GaussianBlur(img, pass1, cv::Size(5, 5), 0);
	cv::fastNlMeansDenoising(pass1, pass1, 21, 7, 21);

	cv::Vec4f mid = segments[0];
	std::vector<cv::Vec4f> opts;
	for(auto l : segments) {
		if(ltick[1] >= l[1] and ltick[3] <= l[3]) {
			opts.push_back(l);
			if(l[3] - l[1] > mid[3] - mid[1]) 
				mid = l;
		}
	}


	std::vector<cv::Vec4f> cull;
	for(const auto& l : opts) {
		if(!(l[1] < ltick[1]-(ltick[3]-ltick[1]))) {
			cull.push_back(l);
		} else if(!(l[3] > ltick[3]+(ltick[3]-ltick[1]))) {
			cull.push_back(l);
		}
	}

	for(auto l : cull) {
		if(std::abs(x_mid - l[0]) < std::abs(x_mid - mid[0]))
			mid = l;
	}

	// equiv of proc_peak() from spectreye.py
	
	int y = mid[1] + (mid[3]-mid[1])/2;

	std::vector<int> ticks;
	for(int x=0; x<pass1.size().width; x++) {
		if(pass1.at<unsigned char>(y, x) > pass1.at<unsigned char>(y, x+1)+1 &&
				pass1.at<unsigned char>(y, x) > pass1.at<unsigned char>(y, x-1)+1) {
			ticks.push_back(x);
		}
	}


	std::vector<int> diffs;
	for(int i=0; i<ticks.size()-1; i++) 
		diffs.push_back(std::abs(ticks[i]-ticks[i+1]));
	
	std::unordered_map<int, int> freq_count;
	for(const auto& d : diffs) 
		freq_count[d]++;

	using ptype = decltype(freq_count)::value_type; // c++11 got me feeling funny
	auto mfreq = std::max_element(freq_count.begin(), freq_count.end(),
			[] (const ptype &x, const ptype &y) {return x.second < y.second;});
	pixel_ratio = std::abs(mfreq->first);


	std::vector<int> iheights;
	int uy, dy;
	for(const auto& l : ticks) {
		uy = y;
	//while(pass1.at<unsigned char>(uy, l) < pass1.at<unsigned char>(uy-1,l)+5)
		while(uy >= ysplit)
			uy--;
		dy = y;
		ysplit = uy;
		while(pass1.at<unsigned char>(dy, l) < pass1.at<unsigned char>(dy+1,l)+5)
			dy++;
		iheights.push_back(dy-uy);
	}

	std::vector<int> opti, heights, locs, dists;
	
	std::priority_queue<std::pair<int, int>> q;
	for(int i=0; i<iheights.size(); ++i) {
		q.push(std::pair<int, int>(iheights[i], i));
	}
	for(int i=0; i<5; ++i) {
		int ki = q.top().second;
		opti.push_back(ki);
		q.pop();
	}

	for(const auto& i : opti) {
		heights.push_back(iheights[i]);
		locs.push_back(ticks[i]);

	}

	for(const auto& l : locs) {
		dists.push_back(std::abs(x_mid - l));
	}
	std::vector<int> dsorted = dists;
	std::sort(dsorted.begin(), dsorted.end());


	auto iter = std::find(dists.begin(), dists.end(), dsorted[0]) - dists.begin();
	auto iter2 = std::find(dists.begin(), dists.end(), dsorted[1]) - dists.begin();

	true_mid = locs[(heights[iter] > heights[iter2]) ? iter : iter2];
	
	cv::rectangle(display, cv::Point(true_mid, ysplit), cv::Point(true_mid, display.size().height), 
			cv::Scalar(255, 255, 0), 2);

	cv::Mat timg = this->ThreshFilter(img);

	std::vector<cv::Rect> boxes = this->OcrEast(timg);

	if(boxes.size() == 0) {
		timg = this->MaskFilter(timg);
		boxes = this->OcrEast(timg);
		if(boxes.size() == 0)
			boxes = this->OcrTess(timg);
	}
	
	if(boxes.size() == 0) {
		printf("failure\n");
		res.status = RC_FAILURE;
		res.dev_type = dtype;
		res.filename = ipath;
		return res;
	}

	int cmpX, botY, bH;
	cv::Rect boxdata;
	for(const auto& rect : boxes) {
		int startX = rect.x;
		int startY = rect.y;
		int endX = rect.x + rect.width;
		int endY = rect.y + rect.height;

		int tpos = startX + (endX-startX)/2;

		if (std::abs(x_mid - tpos) < std::abs(x_mid - cmpX) && endY < ysplit - ((endY-startY)/3)) {
			cmpX = tpos;
			botY = endY;

			bH = std::abs(endY-startY)/2;

			boxdata = cv::Rect(
				std::max(0, startX-this->npadx),
				std::max(0, startY-(this->npady/2)),
				std::min(img.size().width,  (endX-startX)+this->npadx*2),
				std::min(img.size().height, (endY-startY)+(this->npady)+(this->npady/2))
			);
		}
	}
	cv::rectangle(display, boxdata, cv::Scalar(0, 255, 0), 2);

	cv::Vec4f tseg = segments[0];
	for(const auto& l : segments) {
		if(std::abs(cmpX - l[0]) < std::abs(cmpX-tseg[0]) && 
				l[1] > botY && l[1] < ysplit)
			tseg = l;
		
	}
	int tmidy = tseg[1] + ((tseg[3]-tseg[1])/4)*3;
	int tick = tseg[0];
	if(tick > boxdata.x+boxdata.width || tick < boxdata.x)
		tick = boxdata.x + boxdata.width/2; 

	cv::rectangle(display, cv::Point(tick, boxdata.y+boxdata.height), cv::Point(tick, ysplit), cv::Scalar(255, 0, 0), 2);


	int pix_frac = true_mid - tick;
	double dec_frac = ((double)pix_frac/pixel_ratio)*0.01;

	if(boxdata.width == 0) {
		res.status = RC_FAILURE;

		std::cout << Spectreye::DescribeReading(res) << std::endl;

		this->tess->End();
		return res;
	}

	cv::Mat numbox = cv::Mat(timg.clone(), boxdata);
	cv::cvtColor(numbox, numbox, cv::COLOR_GRAY2BGR);
	numbox = this->CLAHEFilter(numbox, 3);
//	cv::morphologyEx(numbox, numbox, cv::MORPH_CLOSE, this->ckernel);
//	cv::threshold(numbox, numbox, 200, 255, cv::THRESH_BINARY);

	this->tess->SetImage((unsigned char*)numbox.data, numbox.size().width, numbox.size().height, 
			numbox.channels(), numbox.step1());
	std::string rawnum = this->tess->GetUTF8Text();
	bool tess2 = false;
	
build_mark: // :-)
	
	std::string nstr;
	for(const auto& n : rawnum) {
		if(std::isdigit(n))
			nstr += n;
		if(n == '\n')
			break;
	}
	if(nstr.length() < 3)
		nstr += ".0";
	else
		nstr.insert(2, ".");
	
	int tickR = 1;
	double mark = std::stod(nstr);

	if(dtype == DT_SHMS) {
		tickR = -1;
		if(mark > SHMS_MAX)
			mark /= 10;
		if(mark < SHMS_MIN)
			mark = 0;
	} else if(dtype == DT_HMS) {
		if(mark > HMS_MAX)
			mark /= 10;
		if(mark < HMS_MIN)
			mark = 0;
	}

	if(mark == 0 && !tess2) {
		std::cout << "rebuild" << std::endl;
		tess2 = true;
		this->tess->SetPageSegMode(tesseract::PageSegMode::PSM_SPARSE_TEXT);
		rawnum = this->tess->GetUTF8Text();
		goto build_mark; // :-)
	}

	double ns1   = mark + (tickR * dec_frac);
	double pow   = std::pow(10.0f, 2);
	double angle = std::round(ns1 * pow)/pow;

	double composite = 0.0;	
	if(enc_angle > 0) {
		double enc_mark = (std::floor((enc_angle*2)+0.5)/2);
		composite = enc_mark + (angle-mark);
	} 

	res.mark = mark;
	res.tick = (tickR * dec_frac);

	if(enc_angle > 0) 
		res.comp_guess = composite;
	if(mark > 0) 
		res.ocr_guess = angle;

	if(res.ocr_guess != 0) {
		if(enc_angle > 0 && res.comp_guess != 0 &&
				std::abs(res.ocr_guess - enc_angle) > MARK_THRESH) {
			res.status = RC_EXCEED;
			res.angle = res.comp_guess;
		} else {
			res.status = RC_SUCCESS;
			res.angle = res.ocr_guess;
		}
	} else if(res.comp_guess != 0) {
		res.status = RC_NOREAD;
		res.angle = res.comp_guess;
	} else {
		res.status = RC_FAILURE;
	}

	if(this->debug) {
		cv::rectangle(display, cv::Point(0, display.size().height-92),
				cv::Point(display.size().width, display.size().height-90),
				cv::Scalar(0, 0, 0), cv::FILLED);

		cv::rectangle(display, cv::Point(0, display.size().height-90), 
				cv::Point(display.size().width, display.size().height),
				cv::Scalar(127, 127, 127), cv::FILLED);
		
		std::stringstream l1, l2;

		l1 << std::fixed << std::setprecision(2) << angle;
		l1 << " degrees. (OCR)";
		cv::putText(display, l1.str(), cv::Point(10, display.size().height-60), this->font, 0.75, 
				cv::Scalar(0, 255, 0), 2);

		if(enc_angle) {
			std::stringstream l;
			l << std::fixed << std::setprecision(2) << composite;
			l << " degrees. (COMP)";

			cv::putText(display, l.str(), 
					cv::Point(display.size().width/2+10, display.size().height-20),
					this->font, 0.75, cv::Scalar(0, 255, 0), 2);
			l2 << std::fixed << std::setprecision(2) << enc_angle;
			l2 << " degrees. (" << ((dtype == DT_SHMS) ? "SHMS" : "HMS") << " ENC)";

			cv::putText(display, l2.str(), cv::Point(10, display.size().height-20), this->font, 0.75,
					cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("final", display);
//		cv::imshow("nb", numbox);
		for(;;) {
			auto key = cv::waitKey(1);
			if(key == 113)
				break;
		}
		cv::destroyAllWindows();
	}
	this->tess->End();

	return res;	
}
