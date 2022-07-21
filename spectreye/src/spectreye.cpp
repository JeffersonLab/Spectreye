// Copyright 2022, Jefferson Science Associates, LLC.
// Subject to the terms in the LICENSE file found in the top-level directory.

/*
 * Spectreye is a tool for automatically determining the angle of the Super High Momentum Spectrometer
 * (SHMS) and the High Momentum Spectrometer (HMS) in JLab's Experimental Hall C. The program uses 
 * computer vision and optical character recognition to determine the angle of the spectrometers from
 * photos of their Vernier calipers. 
 *
 * A non-technical description of the project can be found at 
 * https://docs.google.com/presentation/d/1qKy9npTbnCOFVQCxMHfdYh_vlz-lOZ7rnzxkA6Q_Qw8/
 */


#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <cstdio>
#include <stdlib.h>

#include "include/spectreye.h"

/*
 *  Spectreye is intended to be used by creating a single object, and then using it on multiple image
 *  files. Don't call the constructor for each image, it's a waste of resources and runtime.
 */
Spectreye::Spectreye(int debug) 
{
	// Create morphology elements for repeated use during image filtering.
	this->dkernel = cv::getStructuringElement(cv::MORPH_DILATE, cv::Size(4,4));
	this->okernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1,1));
	this->ckernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2,4));

	// Load models into memory.
	this->net = cv::dnn::readNet(std::string(EAST_PATH));
	this->lsd = cv::createLineSegmentDetector(0);

	// Initialize CLAHE for filtering use.
	this->clahe = cv::createCLAHE();
	this->clahe->setClipLimit(2.0);
	this->clahe->setTilesGridSize(cv::Size(8, 8));

	// Load Tesseract into memory, and set initial configuration. PSM can be changed during run.
	this->tess = new tesseract::TessBaseAPI();
	this->tess->Init(NULL, "eng");
	this->tess->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_BLOCK);
	this->debug = debug;
}

/*
 *  Wrapper for HMS images.
 */
SpectreyeReading Spectreye::GetAngleHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_HMS, path, encoder_angle);
}

/*
 *  Wrapper for SHMS images.
 */
SpectreyeReading Spectreye::GetAngleSHMS(std::string path, double encoder_angle) 
{
	return this->FromFrame(cv::imread(path), DT_SHMS, path, encoder_angle);
}

/*
 *  Utility for extracting the first occurance of a plain-text timestamp out of a file.
 *  These dates are useful for comparing results with encoder data.
 *  There is a lot of room for optimization here.
 */
std::string Spectreye::ExtractTimestamp(std::string path) 
{
	// This can and should be made much more efficient.
	// It currently uses Perl syntax regex which doesn't work on MacOS grep.
	std::string com = "strings " + path + " | grep -P \"(19|20)[\\d]{2,2}\"";

	char buf[128];
	std::string res;

	// Opens file as pipe and writes it to string through 128 byte buffer. 
	// You could probably use libjpeg or something to do this more easily.
	FILE* pipe = popen(com.c_str(), "r");
	if(!pipe)
		return "failed to build timestamp";

	while(!feof(pipe)) {
		if(fgets(buf, 128, pipe) != NULL)
			res += buf;
	}
	pclose(pipe);

	// Encoder datasets seperate dates with '-', but times with ':'.
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

/*
 *  Pretty-print for SpectreyeReading struct. Relies on your terminal having basic color
 *  functionality to make everything nice and fancy looking. Run this in stock xterm if you want
 *  to go blind.
 */
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

	// Changes float position based on signage and other factors to make sure that numbers align.
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

/*
 *	Filter utility that boosts image contrast and reduces shadows, can be efficiently applied 
 *	multiple times by specifying 'passes'. Expects a BGR Mat as input.
 */
cv::Mat Spectreye::CLAHEFilter(cv::Mat frame, int passes) 
{
	cv::Mat img, lab;
	cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
	s
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

/*
 *  Primary filter for OCR. Uses a 'dumb' threshold, and then heavily filters out noise to isolate
 *  v text. Sometimes the threshold will be too low/high, which invalidats the output, and forces the
 *  mask filter to be usd.
 */
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

/*
 *  Alternative filter for OCR which uses a color mask to get rid of everything red. This isn't good 
 *  for reading text, but might actually be better than the threshold for identifying numbers. I have
 *  not tested it as much, so the case may be that it could heavily improve OCR performance with 
 *  some small tweaks.
 */
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

/*
 *  Uses the EAST OCR model to locate bounding boxes for angle marks. Main part of the program that
 *  needs improvement to boost reliability. The EAST model is optimized for 'scenes' where text is
 *  on things like street signs or license plates, so the somewhat abstract environment of the 
 *  Vernier scale is sometimes off-putting to it.
 */
std::vector<cv::Rect> Spectreye::OcrEast(cv::Mat frame) 
{
	std::vector<cv::Mat> outs;
	
	cv::Mat timg;
	cv::cvtColor(frame, timg, cv::COLOR_GRAY2BGR);

	int H = timg.size().height;
	int W = timg.size().width;

	// Images passed to EAST must have be squares with sizes that are multiples of 32.
	int origH = H;
	int newW = 320, newH = 320;
	double rW = (double) W / (double) newW;
	double rH = (double) H / (double) newH;

	cv::resize(timg, timg, cv::Size(newW, newH));
	H = timg.size().height;
	W = timg.size().width;

	// Creates blob using predetermined weights. Don't change them!
	cv::Mat blob = cv::dnn::blobFromImage(
			timg, 1.0, cv::Size(H, W), cv::Scalar(123.68, 116.78, 103.94), true, false); 

	// Runs the model on the blob.
	this->net.setInput(blob);
	this->net.forward(outs, this->layer_names);

	cv::Mat scores   = outs[0];		// Model confidence in each box.
	cv::Mat geometry = outs[1];		// Weird data structure containing box shape and angles.

	std::vector<cv::Rect> rects;	// Final rects to return.
	std::vector<float> confidences;	// Final confidences to rturn.

	int nrows = scores.size[2];
	int ncols = scores.size[3];		// Number of boxes found initially.

	cv::Mat temp = frame.clone();

	for(int y=0; y<nrows; ++y) {
		const float* scores_data = scores.ptr<float>(0, 0, y);
		const float* xdata0 = geometry.ptr<float>(0, 0, y);
		const float* xdata1 = geometry.ptr<float>(0, 1, y);
		const float* xdata2 = geometry.ptr<float>(0, 2, y);
		const float* xdata3 = geometry.ptr<float>(0, 3, y);
		const float* angles = geometry.ptr<float>(0, 4, y);

		for(int x=0; x<ncols; ++x) {

			// Throw out boxes under arbitrary confidence threshold.
			if (scores_data[x] < 0.5) {
				continue;	
			}

			// EAST boxes are represented in polar coordinates on 320x320 image, so they must
			// be resized and converted to Cartesian coordinates for use on the real image.
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
			
			// Make sure that rects don't go off the image vertically.
			if (endY*rH < origH) {

				// Creates a cartesian rectangle resized to our image size.
				cv::Rect rect = cv::Rect(startX*rW, startY*rH, (endX-startX)*rW, (endY-startY)*rH);

				cv::rectangle(temp, rect, cv::Scalar(0, 255, 0), 2);

				// Adds rect to final vector.
				rects.push_back(rect);
				confidences.push_back(scores_data[x]);
			}
		}
	}

	// This tries to perform 'Non-Maximal Suppression', an algorithm which takes boxes with 
	// significant amounts of overlap and combines them. The Python prototype did this very well
	// using an algorithm from the 'imutils' package, but that is not available for C++. This 
	// implementation is less reliable. A better solution would be preferred.
	std::vector<int> indices;
	cv::dnn::NMSBoxes(rects, confidences, 0, 0.5, indices);

	return rects;
}

/*
 *  Tesseract OCR for bounding boxes. Sometimes when EAST fails, Tesseract is able to find the angle
 *  marks. This is currently less reliable than it's Python alternative, as it returns bounding boxes
 *  that are too large horizontally. However, it tends to work where EAST doesn't.
 */
std::vector<cv::Rect> Spectreye::OcrTess(cv::Mat frame) 
{
	std::cout << "Using Tess OCR" << std::endl;
	cv::Mat img, lab;	

	//This should be replaced with a call to CLAHEFilter. It does the exact same thing.
	cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
	cv::cvtColor(frame, lab, cv::COLOR_BGR2Lab);
	
	std::vector<cv::Mat> lplanes(3);
	cv::split(lab, lplanes);
	cv::Mat dst;
	this->clahe->apply(lplanes[0], dst);
	dst.copyTo(lplanes[0]);
	cv::merge(lplanes, lab);
	cv::cvtColor(lab, img, cv::COLOR_Lab2BGR);

	// General denoising and filtering.
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);
	cv::GaussianBlur(img, img, cv::Size(3, 3), 0);
	cv::fastNlMeansDenoising(img, img, 21, 7, 21);

	// Tesseract can't take cv::Mat as an image, so we have to point to the buffer in memory where
	// the image data is stored.
	this->tess->SetImage((unsigned char*)img.data, img.size().width, img.size().height, 
			img.channels(), img.step1());
	this->tess->Recognize(0);
	tesseract::ResultIterator* ri = this->tess->GetIterator();
	tesseract::PageIteratorLevel level = tesseract::RIL_WORD;

	std::vector<cv::Rect> rects;

	// Create bounding boxes just like in EAST. They are thankfully Cartesian.
	if(ri != 0) {
		do {
			float conf = ri->Confidence(level);
			int x1, y1, x2, y2;
			ri->BoundingBox(level, &x1, &y1, &x2, &y2);

			// Throws out rects with low confidence. Note the different confidence level from EAST.
			if(conf > 40 && y2 <= img.size().height/2) {
				rects.push_back(cv::Rect(x1, y1, x2, y2));
			}

		} while (ri->Next(level));
	}

	return rects;
}

/*
 *  Locates the center of the nearest tick by walking left and right from a given point until the 
 *  color is 'delta' points higher than the last pixel. Since the images are grayscale, this point
 *  is most likely the point that the black tick starts returning to the white plate, representing
 *  the center of the tick. Ideally 'ytest' is the vertical center of the tick.
 */
int Spectreye::FindTickCenter(cv::Mat img, int ytest, int xtest, int delta) 
{
	int optl = 0, optr = 0;

	// Walk left until color increases significantly.
	for(int x=xtest-1; x>=1; x--) {
		if(img.at<unsigned char>(ytest, x) > img.at<unsigned char>(ytest, x-1)+delta) {
			optl = x;
			break;
		}
	}

	// Walk right until color increases significantly.
	for(int x=xtest; x<img.size().width; x++) {
		if(img.at<unsigned char>(ytest, x) > img.at<unsigned char>(ytest, x+1)+delta) {
			optr = x;
			break;
		}
	}
	
	// Return x value nearest to starting point.
	return (std::abs(xtest-optl) < std::abs(xtest-optr)) ? optl : optr;	
}

/*
 *  Main Spectreye algorithm. Requires device to determine tick direction, and encoder angle to 
 *  create composite angle guess upon OCR failure. The responsibility of loading the image falls on
 *  the wrapper which calls this metho.
 */
SpectreyeReading Spectreye::FromFrame(
		cv::Mat frame, DeviceType dtype, std::string ipath, double enc_angle)
{
	SpectreyeReading res;

	// C-style strategy for getting the absolute path of the image file.
	const char* icpy = ipath.c_str();
	const char* tpath = const_cast<char*>(icpy);
	ipath = std::string(realpath(tpath, NULL));

	// Grabs timestamp from within iamge file.
	std::string timestamp = this->ExtractTimestamp(ipath);

	// Starts building the return object in case of early failure.
	res.dev_type = dtype;
	res.filename = ipath;
	res.timestamp = timestamp;

	// Absolute image mid-points =/= scale midpoint.
	int x_mid = frame.size().width/2;
	int y_mid = frame.size().height/2;

	// Split frame into two images: 'img' for filtering, and 'display' for showing in debug mode.
	cv::Mat img, display;
	cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
	cv::cvtColor(img, display, cv::COLOR_GRAY2BGR);	
	
	/*
	 *	This section runs the LineSegmentDetector on the images, and reduces its output to lines 
	 *	which are most likely 1/100th angle ticks on the Vernier caliper. Reference images on
	 *	slideshow linked at top of file.
	 */ 
	
	cv::Vec4f ltick, rtick;			 // Two line segments closest to image middle.
	std::vector<cv::Vec4f> segments; // Detected line segments that are probably 1/100th ticks.

	// Filter img for line segment detection.
	cv::Mat pass1;
	cv::fastNlMeansDenoising(img, pass1, 21, 7, 21);
	cv::GaussianBlur(pass1, pass1, cv::Size(5, 5), 0);
	
	// Runs LSD model on filtered image.
	std::vector<cv::Vec4f> lines;
	this->lsd->detect(pass1, lines);
	ltick = lines[0];
	rtick = lines[1];

	// Initial pass to eliminate segments that are outside the bounds of the caliper. 
	for(int i=0; i<lines.size(); i++) {
		cv::Vec4f l = lines[i];

		// Only take lines which are taller than they are wide.
		if(std::abs(l[0] - l[2]) < std::abs(l[1] - l[3])) {

			// Only take lines within the center 10% of the image.
			if(l[1] > y_mid - (0.1 * img.size().height) &&
					l[3] < y_mid + (0.1 * img.size().height)) {
				segments.push_back(l);
				
				// Find the ticks nearest to middle on both the left and right. 
				if(l[0] < x_mid && std::abs(x_mid - l[0]) < std::abs(x_mid - ltick[0])) 
					ltick = l;	
				if(l[0] > x_mid && std::abs(x_mid - l[0]) < std::abs(x_mid - rtick[0])) 
					rtick = l;
			}
		}
	}

	/*
	 *  With the segments reduced to probable caliper ticks, this section attempts to find a tick
	 *  very near the center of the caliper to use as a basis for finding the real center position.
	 */

	int true_mid, ysplit;
	float pixel_ratio;

	// Filter in preparation for walking along x-axis. Reverse order of previous filter. It matters!
	cv::GaussianBlur(img, pass1, cv::Size(5, 5), 0);
	cv::fastNlMeansDenoising(pass1, pass1, 21, 7, 21);

	// Finds all segments above or equal to ltick in the image.
	cv::Vec4f mid = segments[0];	
	std::vector<cv::Vec4f> opts;
	for(auto l : segments) {
		if(ltick[1] >= l[1] and ltick[3] <= l[3]) {
			opts.push_back(l);
			if(l[3] - l[1] > mid[3] - mid[1]) 
				mid = l;	// Sets initial guess to tallest tick in opts
		}
	}

	// Gets rid of lines significantly larger than ltick, as they probably aren't caliper ticks.
	std::vector<cv::Vec4f> cull;
	for(const auto& l : opts) {
		if(!(l[1] < ltick[1]-(ltick[3]-ltick[1]))) {
			cull.push_back(l);
		} else if(!(l[3] > ltick[3]+(ltick[3]-ltick[1]))) {
			cull.push_back(l);
		}
	}

	// Sets guess to segment in cull clossest to static image midpoint.
	for(auto l : cull) {
		if(std::abs(x_mid - l[0]) < std::abs(x_mid - mid[0]))
			mid = l;
	}
	
	int y = mid[1] + (mid[3]-mid[1])/2; // Sets y-value for walking to center of current mid guess.

	/*
	 *  With a position identified near the center of the caliper, the image can be walked left 
	 *  and right, to locate the peaks of as many caliper ticks as possible to use as references for
	 *  determining the ratio of pixels/degrees.
	 */

	// Walks entire image at y=y and finds 'peaks' where color change is > 2
	std::vector<int> ticks;
	for(int x=0; x<pass1.size().width; x++) {
		if(pass1.at<unsigned char>(y, x) > pass1.at<unsigned char>(y, x+1)+1 &&
				pass1.at<unsigned char>(y, x) > pass1.at<unsigned char>(y, x-1)+1) {
			ticks.push_back(x);
		}
	}
	
	// Gets the positional difference between each identified peak. 
	std::vector<int> diffs;
	for(int i=0; i<ticks.size()-1; i++) 
		diffs.push_back(std::abs(ticks[i]-ticks[i+1]));
	
	// Creates amap of each difference value and the number of times it occurs
	std::unordered_map<int, int> freq_count;
	for(const auto& d : diffs) 
		freq_count[d]++;

	// This is really ugly, but it essentially finds the difference with the highest occurance
	// within freq_count, and sets the final pixel/degree ratio to it.
	using ptype = decltype(freq_count)::value_type; // c++11 got me feeling funny
	auto mfreq = std::max_element(freq_count.begin(), freq_count.end(),
			[] (const ptype &x, const ptype &y) {return x.second < y.second;});
	pixel_ratio = std::abs(mfreq->first);

	// For each tick, we walk the image vertically to find it's height. We know that the tick ends
	// when the color delta shoots up. This strategy is much better for finding lines than just
	// relying on the LineSegmentDetector because it can find the true center of ticks based on 
	// their gradient rather than just looking for hard edges, which lead to inaccuracies.
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

	/*
	 *	This section compares the heights and distances from center of each tick to narrow down
	 *	possible midpoints, and eventually selects the best possible guess.
	 */

	// Creates vectors for the heights of the next round of ticks, 
	// their locations, and their distance from the image center.
	std::vector<int> opti, heights, locs, dists;
	
	// A priority queue is most likely overkill for this task.
	// This stores the indices of the 5 tallest ticks in 'opti'.
	std::priority_queue<std::pair<int, int>> q;		
	for(int i=0; i<iheights.size(); ++i) {
		q.push(std::pair<int, int>(iheights[i], i));
	}
	for(int i=0; i<5; ++i) {
		int ki = q.top().second;
		opti.push_back(ki);
		q.pop();
	}

	// Adds the heights and locations of the 5 tallest ticks based on their indices from opti.
	for(const auto& i : opti) {
		heights.push_back(iheights[i]);
		locs.push_back(ticks[i]);

	}
	// Adds corresponding distances for ease of use.
	for(const auto& l : locs) {
		dists.push_back(std::abs(x_mid - l));
	}

	// Clones the distances and sorts them.
	std::vector<int> dsorted = dists;
	std::sort(dsorted.begin(), dsorted.end());

	// Finds the indices of the two twicks closest two the middle, by using find() on the value of 
	// the first two elements of dsorted. 
	auto iter = std::find(dists.begin(), dists.end(), dsorted[0]) - dists.begin();
	auto iter2 = std::find(dists.begin(), dists.end(), dsorted[1]) - dists.begin();

	// Finally finds the true center of the caliper by using the taller of the two options. This
	// takes advantage of the fact that the center caliper tick is slightly taller than the others.
	true_mid = locs[(heights[iter] > heights[iter2]) ? iter : iter2];
	
	cv::rectangle(display, cv::Point(true_mid, ysplit), cv::Point(true_mid, display.size().height), 
			cv::Scalar(255, 255, 0), 2);

	/*
	 *  With the caliper midpoint and pixel/degree ratio indentified, the algorithm begins the OCR
	 *  process, using multiple different filtering and detection strategies to read the angle mark.
	 */ 

	// Starts OCR process with threshold-based filter.
	cv::Mat timg = this->ThreshFilter(img);

	// First OCR attempt uses EAST for its high bounding box accuracy when successful.
	std::vector<cv::Rect> boxes = this->OcrEast(timg);

	// If EAST fails, try the mask filter and run it again. If it fails again, try Tesseract.
	if(boxes.size() == 0) {
		timg = this->MaskFilter(timg);
		boxes = this->OcrEast(timg);
		if(boxes.size() == 0)
			boxes = this->OcrTess(timg);
	}
	
	// If we still can't find anything, give up and return a failure.
	if(boxes.size() == 0) {
		printf("failure\n");
		res.status = RC_FAILURE;
		res.dev_type = dtype;
		res.filename = ipath;
		return res;
	}

	/*
	 *  Once bounding boxes are identified, the one most likely to be the angle marking is selected
	 *  based on its position in the image.
	 */

	
	int cmpX; // Center of box nearest to x-center.
	int botY  // Bottom of selected rect.
	int bH;   // Selected rect height/2.

	cv::Rect boxdata;
	for(const auto& rect : boxes) {
		int startX = rect.x;
		int startY = rect.y;
		int endX = rect.x + rect.width;
		int endY = rect.y + rect.height;

		int tpos = startX + (endX-startX)/2;

		// Take the bounding box closest to the center which is also a certain margin above the top
		// of the middle tick. If the OCR worked correctly, this will be the bounding box of the
		// angle mark.
		if (std::abs(x_mid - tpos) < std::abs(x_mid - cmpX) && endY < ysplit - ((endY-startY)/3)) {
			cmpX = tpos;
			botY = endY;

			bH = std::abs(endY-startY)/2;

			// Add padding to bounding box to make sure that no numbers get cut off during reading.
			boxdata = cv::Rect(
				std::max(0, startX-this->npadx),
				std::max(0, startY-(this->npady/2)),
				std::min(img.size().width,  (endX-startX)+this->npadx*2),
				std::min(img.size().height, (endY-startY)+(this->npady)+(this->npady/2))
			);
		}
	}
	cv::rectangle(display, boxdata, cv::Scalar(0, 255, 0), 2);

	// Because of how simple the 'big' ticks are, we can use the LSD segments to find 
	// the location of the tick closest to the angle mark.
	cv::Vec4f tseg = segments[0];
	for(const auto& l : segments) {
		if(std::abs(cmpX - l[0]) < std::abs(cmpX-tseg[0]) && 
				l[1] > botY && l[1] < ysplit)
			tseg = l;
		
	}
	int tmidy = tseg[1] + ((tseg[3]-tseg[1])/4)*3;
	int tick = tseg[0]; // Set tick x position to the top of the selected segment.

	// If this goes wrong and the selected position is outside the the box, we approximate
	// by setting the position to the center of the bounding box, which is a close enough
	// approximation, but might affect accuracy if the numbers are far from the center of the image.
	if(tick > boxdata.x+boxdata.width || tick < boxdata.x)
		tick = boxdata.x + boxdata.width/2; 

	cv::rectangle(display, cv::Point(tick, boxdata.y+boxdata.height), cv::Point(tick, ysplit), cv::Scalar(255, 0, 0), 2);


	int pix_frac = true_mid - tick; // Distance in pixels from caliper center to angle mark
									
	// Final result of all the numerical processing. This number is the angle (in degrees) between
	// the angle marking and the center of the caliper. This will be added to the OCR result or
	// the encoder mark to find the final angle of the image.
	double dec_frac = ((double)pix_frac/pixel_ratio)*0.01;	

	// Sometimes the OCR fails and returns a box with a width of 0. 
	// We return a failure if this happens.
	if(boxdata.width == 0) {
		res.status = RC_FAILURE;

		std::cout << Spectreye::DescribeReading(res) << std::endl;

		this->tess->End();
		return res;
	}

	/*
	 *	To get the final angle, we need to know what the angle mark actually is so we can add it 
	 *	to the fractional angle we just calculated. The ideal way to do this is to read it with 
	 *	Tesseract, so that we don't have to rely on the encoder mark.
	 */

	// Crop the image to the bounding box of the angle mark and boost contrast for reading.
	cv::Mat numbox = cv::Mat(timg.clone(), boxdata);
	cv::cvtColor(numbox, numbox, cv::COLOR_GRAY2BGR);
	numbox = this->CLAHEFilter(numbox, 3);
//	cv::morphologyEx(numbox, numbox, cv::MORPH_CLOSE, this->ckernel);
//	cv::threshold(numbox, numbox, 200, 255, cv::THRESH_BINARY);
	
	// Pass the image as a buffer in memory to Tesseract. Would be nice if it took cv::Mat.
	this->tess->SetImage((unsigned char*)numbox.data, numbox.size().width, numbox.size().height, 
			numbox.channels(), numbox.step1());
	std::string rawnum = this->tess->GetUTF8Text();	// Tesseract text output.
	bool tess2 = false;
	
	// Once the Tesseract output is acquired, it needs to be interpreted into a viable double.
	// This process has a label/goto so that it can be repeated a few times with different 
	// Tesseract configurations in case the intial read fails.

build_mark: // :-)
	
	// Cull all non-digit characters from string, and only take numbers from the 
	// first line containing any digits.
	std::string nstr;
	for(const auto& n : rawnum) {
		if(std::isdigit(n))
			nstr += n;
		if(n == '\n')
			break;
	}

	// Add .0 if numer is only 2 digits, otherwise, add a decimal point after 2 digits.
	if(nstr.length() < 3)
		nstr += ".0";
	else
		nstr.insert(2, ".");
	
	int tickR = 1;	// 1 if HMS, -1 if SHMS.
	double mark = std::stod(nstr);	// Alterted string converted to double. This shouldn't fail.

	// Additional operations/checks based on device type.
	if(dtype == DT_SHMS) {
		// Make tick angle negative because SHMS is right->left.
		tickR = -1;		
		// Decimal points aren't recognized, so if the angle is >35 we add a '.'
		if(mark > SHMS_MAX)
			mark /= 10;	
		// If the number is under the minimum value as it usually will be if there is no read,
		// it is set to 0 to indicate a failure.
		if(mark < SHMS_MIN)
			mark = 0;	
	} else if(dtype == DT_HMS) {
		if(mark > HMS_MAX)
			mark /= 10;	// Same check as SHMS	
		if(mark < HMS_MIN)
			mark = 0;	// Same failure condition as SHMS
	}

	// If the Tesseract read fails, change some settings and jump back up to build_mark.
	// This is hopefully a safe usage of goto.
	if(mark == 0 && !tess2) {
		std::cout << "rebuild" << std::endl;
		tess2 = true;
		this->tess->SetPageSegMode(tesseract::PageSegMode::PSM_SPARSE_TEXT);
		rawnum = this->tess->GetUTF8Text();
		goto build_mark; // :-)
	}

	// Round angle to 2 decimal points.
	double ns1   = mark + (tickR * dec_frac);
	double pow   = std::pow(10.0f, 2);
	double angle = std::round(ns1 * pow)/pow;

	/*
	 *  Once an OCR angle mark is determined, we need to decide whether or not to rely on it.
	 *  If the program detects a probable issue with the mark reading, it will use a 'composite'
	 *  angle guess. The composite guess uses the encoder value rounded to the nearest 0.5 to 
	 *  build an angle guess, instead of relying on the OCR-read number. This relies on the 
	 *  assumption that the encoder won't be more than 0.5 degrees off, which it hasn't been in 
	 *  the data which I have been provided with.
	 */

	double composite = 0.0;	

	// Only build composite guess if encoder angle exists.
	if(enc_angle > 0) {
		double enc_mark = (std::floor((enc_angle*2)+0.5)/2); // Round encoder angle to 0.5
		composite = enc_mark + (angle-mark);
	} 

	// Start building result object.
	res.mark = mark;
	res.tick = (tickR * dec_frac);

	// Set composite and OCR guesses in result.
	if(enc_angle > 0) 
		res.comp_guess = composite;
	if(mark > 0) 
		res.ocr_guess = angle;

	// Decide the final result of the algorithm.
	if(res.ocr_guess != 0) {
		// If the difference between the encoder angle and the OCR guess is more than the 
		// threshold set in the header, we decide that the OCR was probably wrong and instead
		// return the composite guess. Otherwise the OCR guess is chosen.
		if(enc_angle > 0 && res.comp_guess != 0 &&
				std::abs(res.ocr_guess - enc_angle) > MARK_THRESH) {
			res.status = RC_EXCEED;
			res.angle = res.comp_guess;
		} else {
			res.status = RC_SUCCESS;
			res.angle = res.ocr_guess;
		}
	} else if(res.comp_guess != 0) {
		// If there is no OCR guess but we have encoder data, a "NOREAD" condition will
		// be returned along with the composite guess. This is not a failure, as the ticks
		// should still be accurate.
		res.status = RC_NOREAD;
		res.angle = res.comp_guess;
	} else {
		// If both guesses are set to 0, the program has failed.
		res.status = RC_FAILURE;
	}

	/*
	 *  If the Spectreye object is created in debug mode, an image will be created which
	 *  highlights various elements of the image, and displays each guess.
	 */

	if(this->debug) {
		//Draw line on top of data box
		cv::rectangle(display, cv::Point(0, display.size().height-92),
				cv::Point(display.size().width, display.size().height-90),
				cv::Scalar(0, 0, 0), cv::FILLED);
		//Draw data box
		cv::rectangle(display, cv::Point(0, display.size().height-90), 
				cv::Point(display.size().width, display.size().height),
				cv::Scalar(127, 127, 127), cv::FILLED);
		
		// Creates strings out of each guess by converting them via stringstream.
		// There is probably a more elegant way to do this.
		std::stringstream l1, l2;

		// Print OCR guess.
		l1 << std::fixed << std::setprecision(2) << angle;
		l1 << " degrees. (OCR)";
		cv::putText(display, l1.str(), cv::Point(10, display.size().height-60), this->font, 0.75, 
				cv::Scalar(0, 255, 0), 2);

		// If encoder data exists, print the composite guess.
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

		// Display the final window
		cv::imshow("final", display);
//		cv::imshow("nb", numbox);
		for(;;) {
			auto key = cv::waitKey(1);
			if(key == 113)	// Kill the window is 'q' is pressed.
				break;
		}
		cv::destroyAllWindows();
	}

	return res;	
}

/*
 *  This is required so that Tesseract is only created once for multiple FromFrame() passes.
 *  Might become more useful if other heap allocated data/models are employed down the line.
 */
void Spectreye::Destroy() {
	this->tess->End();
}
