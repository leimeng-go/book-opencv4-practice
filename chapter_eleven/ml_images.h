#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class MLoperatorsDemo {
public:
	void kmeans_segmentation_demo(Mat &image);
	void mainColorComponents(Mat &image);
	void knn_digit_train(Mat &image);
	void knn_digit_test();
	void svm_digit_train(Mat &image);
	void svm_digit_test();
	void get_hog_descriptor(Mat &image, vector<float> &desc);
	void train_ele_watch(std::string positive_dir, std::string negative_dir);
	void hog_svm_detector_demo(Mat &image);
};