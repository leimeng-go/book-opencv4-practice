#include <binary_ops.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";

void BinaryDemo::binary_methods_demo(Mat &image) {
	Mat binary;
	threshold(image, binary, 127, 255, THRESH_BINARY);
	imshow("二值化", binary);
	threshold(image, binary, 127, 255, THRESH_BINARY_INV);
	imshow("二值化反", binary);
	threshold(image, binary, 127, 255, THRESH_TRUNC);
	imshow("阈值截断", binary);
	threshold(image, binary, 127, 255, THRESH_TOZERO);
	imshow("阈值取零", binary);
	threshold(image, binary, 127, 255, THRESH_TOZERO_INV);
	imshow("阈值取零反", binary);
}

void BinaryDemo::global_binary_demo(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	double t = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	std::cout << "threshold value : " << t << std::endl;
	imshow("大津法二值化", binary);

	t = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_TRIANGLE);
	std::cout << "threshold value : " << t << std::endl;
	imshow("三角法二值化", binary);
}

void BinaryDemo::ada_binary_demo(Mat &image) {
	Mat binary;
	adaptiveThreshold(image, binary, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 10);
	imshow("C均值模糊自适应", binary);

	adaptiveThreshold(image, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 10);
	imshow("高斯模糊自适应", binary);
}

void BinaryDemo::noise_and_binary(Mat &image) {
	Mat binary;
	threshold(image, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("二值化", binary);

	Mat denoise_img;
	GaussianBlur(image, denoise_img, Size(5, 5), 0, 0);
	threshold(denoise_img, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("高斯去噪预处理+二值化", binary);

	bilateralFilter(image, denoise_img, 0, 100, 10);
	threshold(denoise_img, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("双边滤波预处理+二值化", binary); 
}

void BinaryDemo::inrange_binary(Mat &image) {
	Mat hsv, mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(20, 43, 46), Scalar(180, 255, 255), mask);
	imshow("区域mask", mask);
	Mat result;
	bitwise_and(image, image, result, mask);
	imshow("区域提取", result);
}

int main(int argc, char** argv) {
	Mat image = imread(rootdir + "coins.jpg", cv::IMREAD_GRAYSCALE);
	imshow("输入图像", image);
	BinaryDemo bd;
	bd.noise_and_binary(image);
	waitKey(0);
	return 0;
}