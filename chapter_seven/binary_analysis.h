#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class BinaryAnalysis {
public:
	void binary_ops(Mat &image);
	void connected_component_demo(Mat &image);
	void find_contours_demo(Mat &image);
	void contours_analysis_demo(Mat &image);
	void contours_fitness_demo(Mat &image);
	void contours_apprv_demo(Mat &image);
	void contours_attrs_demo(Mat &image);
	void hough_line_demo(Mat &image);
	void hough_circle_demo(Mat &image);
	void inner_extenerl_circle_demo(Mat &image);
	void contour_match_demo(Mat &image);
	void max_contour_demo(Mat &image);
	void convex_demo(Mat &image);
};