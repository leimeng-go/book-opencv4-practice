#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class  defectDetection {
public:
	void basic_blob_detection(Mat &image);
	void basic_flaw_detection(Mat &image);
	void hard_blob_detection(Mat &image);
	void hard_flaw_detection(Mat &image);
	void multiple_defects_detection(Mat &image);
	void resnet_surface_detection(Mat &image);
	void segnet_surface_detection(Mat &image);
};