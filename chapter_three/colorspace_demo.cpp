#include <opencv2/opencv.hpp>
#include <ColorSpaceDemo.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.5.4/opencv/sources/samples/data/";

void ColorSpaceDemo::bgr2rgb(Mat &image) {
	Mat dst;
	cvtColor(image, dst, COLOR_BGR2RGB);
	imshow("result", dst);
}

void ColorSpaceDemo::color_range_demo(Mat &image) {
	// RGB to HSV
	Mat hsv;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	imshow("hsv", hsv);

	// RGB to LAB
	Mat lab;
	cvtColor(image, lab, COLOR_BGR2Lab);
	imshow("lab", lab);

	// 提取前景对象
	Mat mask;
	inRange(hsv, Scalar(35, 43, 46), Scalar(99, 255, 255), mask);
	imshow("mask", mask);

	Mat dst;
	bitwise_not(mask, mask);
	bitwise_and(image, image, dst, mask);
	imshow("dst", dst);
}

int main(int argc, char** argv) {
	Mat src = imread(rootdir + "baboon.jpg");
	imshow("input", src);
	// RGB转换
	ColorSpaceDemo cs;
	cs.bgr2rgb(src);
	waitKey(0);

	// 色彩空间转换与应用
	src = imread(rootdir + "green.jpg");
	imshow("input", src);
	cs.color_range_demo(src);

	waitKey(0);
	return 0;
}