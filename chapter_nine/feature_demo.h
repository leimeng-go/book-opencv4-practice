#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

class FeatureVectorOps {
public:
	void pyramid_demo(Mat &image);
	void pyramid_blend_demo(Mat &apple, Mat &orange);
	void harris_demo(Mat &image);
	void shi_tomas_demo(Mat &image);
	void corners_sub_pixels_demo(Mat &image);
	void hog_feature_demo(Mat &image);
	void hog_detect_demo(Mat &image);
	void orb_detect_demo(Mat &image);
	void orb_match_demo(Mat &box, Mat &box_in_scene);
	void find_known_object(Mat &book, Mat &book_on_desk);
};