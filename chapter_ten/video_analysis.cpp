#include <video_analysis.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";

void VideoAnalysisOp::color_based_tracking(std::string videoFilePath) {
	VideoCapture capture;
	bool ret = capture.open(videoFilePath);
	if (!ret) {
		std::cout << "could not open the video file..." << std::endl;
		return;
	}
	Mat frame, hsv, mask;
	Mat se = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	while (true) {
		capture.read(frame);
		if (frame.empty()) {
			break;
		}
		cvtColor(frame, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(30, 0, 245), Scalar(180, 10, 255), mask);
		morphologyEx(mask, mask, MORPH_CLOSE, se);
		findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
		for (size_t t = 0; t < contours.size(); t++) {
			Rect box = boundingRect(contours[t]);
			rectangle(frame, box, Scalar(0, 0, 255), 2, 8, 0);
			if (contours[t].size() > 5) {
				RotatedRect rrt = fitEllipse(contours[t]);
				circle(frame, rrt.center, 3, Scalar(255, 0, 255), 2, 8, 0);
			}
		}
		imshow("frame", frame);
		waitKey(1);
	}
	capture.release();
}

void VideoAnalysisOp::background_analysis_demo(std::string videoFilePath) {
	VideoCapture capture(videoFilePath);

	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("mask", WINDOW_AUTOSIZE);

	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(500, 1000, false);
	Mat frame, mask, back_img;
	while (true) {
		const int64 start = getTickCount();
		bool ret = capture.read(frame);
		if (!ret) break;
		pMOG2->apply(frame, mask);
		pMOG2->getBackgroundImage(back_img);
		imshow("input", frame);
		imshow("mask", mask);
		imshow("back ground image", back_img);
		char c = waitKey(1);
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		std::cout << "FPS : " << fps << std::endl;
		if (c == 27) {
			break;
		}
	}
}

void VideoAnalysisOp::frame_diff_analysis_demo(std::string videoFilePath) {
	VideoCapture capture(videoFilePath);

	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return;
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("result", WINDOW_AUTOSIZE);

	Mat preFrame, preGray;
	capture.read(preFrame);
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	GaussianBlur(preGray, preGray, Size(0, 0), 15);
	Mat binary;
	Mat frame, gray;
	Mat k = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		GaussianBlur(gray, gray, Size(0, 0), 15);
		subtract(gray, preGray, binary);
		threshold(binary, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
		morphologyEx(binary, binary, MORPH_OPEN, k);
		imshow("input", frame);
		imshow("result", binary);

		gray.copyTo(preGray);
		char c = waitKey(5);
		if (c == 27) {
			break;
		}
	}
}

void draw_goodFeatures(Mat &image, vector<Point2f> goodFeatures) {
	for (size_t t = 0; t < goodFeatures.size(); t++) {
		circle(image, goodFeatures[t], 2, Scalar(0, 255, 0), 2, 8, 0);
	}
}

void draw_lines(Mat &image, vector<Point2f> pt1, vector<Point2f> pt2) {
	RNG rng(12345);
	vector<Scalar> color_lut;
	if (color_lut.size() < pt1.size()) {
		for (size_t t = 0; t < pt1.size(); t++) {
			color_lut.push_back(Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)));
		}
	}
	for (size_t t = 0; t < pt1.size(); t++) {
		line(image, pt1[t], pt2[t], color_lut[t], 2, 8, 0);
	}
}

void VideoAnalysisOp::klt_tracking_demo(std::string videoFilePath) {
	VideoCapture capture;
	capture.open(videoFilePath);
	vector<Point2f> featurePoints;
	double qualityLevel = 0.01;
	int minDistance = 10;
	int blockSize = 3;
	bool useHarrisDetector = false;
	double k = 0.04;
	int maxCorners = 5000;
	Mat frame, gray;
	vector<Point2f> pts[2];
	vector<Point2f> initPoints;
	vector<uchar> status;
	vector<float> err;
	TermCriteria criteria = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 30, 0.01);
	double derivlambda = 0.5;
	int flags = 0;

	// detect first frame and find corners in it
	Mat old_frame, old_gray;
	capture.read(old_frame);
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
	goodFeaturesToTrack(old_gray, featurePoints, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
	initPoints.insert(initPoints.end(), featurePoints.begin(), featurePoints.end());
	pts[0].insert(pts[0].end(), featurePoints.begin(), featurePoints.end());
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	Mat result = Mat::zeros(Size(width * 2, height), CV_8UC3);
	Rect roi(0, 0, width, height);
	while (true) {
		bool ret = capture.read(frame);
		if (!ret) break;
		imshow("frame", frame);
		roi.x = 0;
		frame.copyTo(result(roi));
		cvtColor(frame, gray, COLOR_BGR2GRAY);

		// calculate optical flow
		calcOpticalFlowPyrLK(old_gray, gray, pts[0], pts[1], status, err, Size(31, 31), 3, criteria, derivlambda, flags);
		size_t i, k;
		for (i = k = 0; i < pts[1].size(); i++)
		{
			// 距离与状态测量
			double dist = abs(pts[0][i].x - pts[1][i].x) + abs(pts[0][i].y - pts[1][i].y);
			if (status[i] && dist > 2) {
				pts[0][k] = pts[0][i];
				initPoints[k] = initPoints[i];
				pts[1][k++] = pts[1][i];
				circle(frame, pts[1][i], 3, Scalar(0, 255, 0), -1, 8);
			}
		}
		// resize 有用特征点
		pts[1].resize(k);
		pts[0].resize(k);
		initPoints.resize(k);
		// 绘制跟踪轨迹
		draw_lines(frame, initPoints, pts[1]);
		imshow("result", frame);
		roi.x = width;
		frame.copyTo(result(roi));
		char c = waitKey(50);
		if (c == 27) {
			break;
		}

		// update old
		std::swap(pts[1], pts[0]);
		cv::swap(old_gray, gray);

		// need to re-init
		if (initPoints.size() < 40) {
			goodFeaturesToTrack(old_gray, featurePoints, maxCorners, qualityLevel, minDistance, Mat(), blockSize, useHarrisDetector, k);
			initPoints.insert(initPoints.end(), featurePoints.begin(), featurePoints.end());
			pts[0].insert(pts[0].end(), featurePoints.begin(), featurePoints.end());
			printf("total feature points : %d\n", pts[0].size());
		}
	}
}

void VideoAnalysisOp::fb_opticalflow_demo(std::string videoFilePath) {
	VideoCapture capture;
	capture.open(videoFilePath);
	Mat preFrame, preGray;
	capture.read(preFrame);
	cvtColor(preFrame, preGray, COLOR_BGR2GRAY);
	Mat hsv = Mat::zeros(preFrame.size(), preFrame.type());
	Mat frame, gray;
	Mat_<Point2f> flow;
	vector<Mat> mv;
	split(hsv, mv);
	Mat mag = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ang = Mat::zeros(hsv.size(), CV_32FC1);
	Mat xpts = Mat::zeros(hsv.size(), CV_32FC1);
	Mat ypts = Mat::zeros(hsv.size(), CV_32FC1);
	while (true) {
		int64 start = cv::getTickCount();
		bool ret = capture.read(frame);
		if (!ret) break;
		imshow("frame", frame);
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		calcOpticalFlowFarneback(preGray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
		for (int row = 0; row < flow.rows; row++)
		{
			for (int col = 0; col < flow.cols; col++)
			{
				const Point2f& flow_xy = flow.at<Point2f>(row, col);
				xpts.at<float>(row, col) = flow_xy.x;
				ypts.at<float>(row, col) = flow_xy.y;
			}
		}
		cartToPolar(xpts, ypts, mag, ang);
		ang = ang * 180.0 / CV_PI / 2.0;
		normalize(mag, mag, 0, 255, NORM_MINMAX);
		convertScaleAbs(mag, mag);
		convertScaleAbs(ang, ang);
		mv[0] = ang;
		mv[1] = Scalar(255);
		mv[2] = mag;
		merge(mv, hsv);
		Mat bgr;
		cvtColor(hsv, bgr, COLOR_HSV2BGR);
		double fps = cv::getTickFrequency() / (cv::getTickCount() - start);
		putText(bgr, format("FPS : %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("result", bgr);
		int ch = waitKey(1);
		if (ch == 27) {
			break;
		}
	}
}

void VideoAnalysisOp::mean_shift_demo(std::string videoFilePath) {
	Mat image;
	bool selectObject = false;
	int trackObject = 0;
	Point origin;

	VideoCapture cap(videoFilePath);
	Rect trackWindow;
	int hsize = 16;
	float hranges[] = { 0,180 };
	const float* phranges = hranges;
	namedWindow("MeanShift Demo", WINDOW_AUTOSIZE);

	Mat frame, hsv, hue, mask, hist, backproj;
	bool paused = false;
	cap.read(frame);
	Rect selection = selectROI("MeanShift Demo", frame, true, false);

	while (true)
	{
		bool ret = cap.read(frame);
		if (!ret) break;
		frame.copyTo(image);

		cvtColor(image, hsv, COLOR_BGR2HSV);
		inRange(hsv, Scalar(26, 43, 46), Scalar(34, 255, 255), mask);
		int ch[] = { 0, 0 };
		hue.create(hsv.size(), hsv.depth());
		mixChannels(&hsv, 1, &hue, 1, ch, 1);

		if (trackObject <= 0)
		{
			// 建立搜索窗口与ROI区域直方图信息
			Mat roi(hue, selection), maskroi(mask, selection);
			calcHist(&roi, 1, 0, maskroi, hist, 1, &hsize, &phranges);
			normalize(hist, hist, 0, 255, NORM_MINMAX);

			trackWindow = selection;
			trackObject = 1;
		}

		// 反向投影
		calcBackProject(&hue, 1, 0, hist, backproj, &phranges);
		backproj &= mask;

		// 均值迁移
		meanShift(backproj, trackWindow, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
		rectangle(image, trackWindow, Scalar(0, 0, 255), 3, LINE_AA);

		imshow("MeanShift Demo", image);
		char c = (char)waitKey(50);
		if (c == 27)
			break;
	}
}

int main(int argc, char** argv) {
	std::string videoFilePath = rootdir + "vtest.avi";
	VideoAnalysisOp analyizer;
	analyizer.fb_opticalflow_demo(videoFilePath);
}
