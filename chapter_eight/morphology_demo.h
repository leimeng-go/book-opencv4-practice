#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class MorphologyOpDemo {
public:
	void dilate_erode_demo(Mat &image);
	void open_close_demo(Mat &image);
	void gradient_demo(Mat &image);
	void gradient_edges(Mat &image);
	void hats_demo(Mat &image);
	void hitandmiss_demo(Mat &image);
	void hvlines_demo(Mat &image);
	void cross_demo(Mat &image);
	void distance_demo(Mat &image);
	void wateshed_demo(Mat &image);
};