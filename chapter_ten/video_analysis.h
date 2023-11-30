#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class VideoAnalysisOp {
public:
	void color_based_tracking(std::string videoFilePath);
	void background_analysis_demo(std::string videoFilePath);
	void frame_diff_analysis_demo(std::string videoFilePath);
	void klt_tracking_demo(std::string videoFilePath);
	void fb_opticalflow_demo(std::string videoFilePath);
	void mean_shift_demo(std::string videoFilePath);
};