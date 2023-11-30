#pragma once
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;
 
class OpenVINOInferenceModels {
public:
	void image_classification_demo(Mat &image);
	void unet_demo(Mat &image);
	void yolov5_demo();
	void landmark_demo(std::string landmark_model_path);
	void setup_test();
};