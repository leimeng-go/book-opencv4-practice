#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudabgsegm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudafilters.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace cv::dnn;
using namespace std;

class CUDASpeedUpDemo {
public:
	void queryDeviceInfo();
	void videoAnalysis();
	void epf();
	void faceDetection();
};