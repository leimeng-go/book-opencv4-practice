#pragma once

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

class DeepNeuralNetOps {
public:
	void image_classification(Mat &image);
	void ssd_demo(Mat &image);
	void faster_rcnn_demo(Mat &image);
	void yolo_demo(Mat &image);
	void enet_demo(Mat &image);
	void style_transfer_demo(Mat &image);
	void text_detection_demo(Mat &image);
	void face_detection_demo(Mat &image, bool tf);
	void cam_face_detection_demo(bool tf);
};