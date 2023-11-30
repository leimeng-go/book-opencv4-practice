#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class YOLOv5Detector {
public:
	void yolov5_infer(std::string model_path, std::string labels_file);
	void camel_elephant_infer(std::string model_path, std::string labels_file);
};