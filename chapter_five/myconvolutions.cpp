#include <opencv2/opencv.hpp>
#include <convolution_ops.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";

void ConvolutionDemo::custom_filter_demo(Mat &image) {
	Mat k1 = Mat::ones(Size(25, 25), CV_32FC1);
	k1 = k1 / (25 * 25);
	std::cout << k1 << std::endl;
	Mat result25x25;
	filter2D(image, result25x25, -1, k1, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("�Զ���ģ��-25x25", result25x25);

	Mat k2 = Mat::ones(Size(25, 1), CV_32FC1);
	k2 = k2 / (25 * 1);
	std::cout << k2 << std::endl;
	Mat result25x1;
	filter2D(image, result25x1, -1, k2, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("�Զ���ˮƽģ��-25x1", result25x1);

	Mat k3= Mat::ones(Size(1, 25), CV_32FC1);
	k3 = k3 / (1 * 25);
	std::cout << k3 << std::endl;
	Mat result1x25;
	filter2D(image, result1x25, -1, k3, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("�Զ��崹ֱģ��-1x25", result1x25);

	Mat k4 = Mat::eye(Size(25, 25), CV_32FC1);
	k4 = k4 / (25);
	std::cout << k4 << std::endl;
	Mat result25;
	filter2D(image, result25, -1, k4, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("�Զ���Խ�ģ��", result25);
}

void ConvolutionDemo::conv_demo(Mat &image) {
	int w = image.cols;
	int h = image.rows;
	Mat result = image.clone();
	for (int row = 1; row < h - 1; row++) {
		for (int col = 1; col < w - 1; col++) {
			int sum_b = image.at<Vec3b>(row, col)[0] + image.at<Vec3b>(row, col - 1)[0] + image.at<Vec3b>(row, col + 1)[0]
				+ image.at<Vec3b>(row - 1, col)[0] + image.at<Vec3b>(row - 1, col - 1)[0] + image.at<Vec3b>(row - 1, col + 1)[0]
				+ image.at<Vec3b>(row+1, col)[0] + image.at<Vec3b>(row+1, col - 1)[0] + image.at<Vec3b>(row+1, col + 1)[0];
			
			int sum_g = image.at<Vec3b>(row, col)[1] + image.at<Vec3b>(row, col - 1)[1] + image.at<Vec3b>(row, col + 1)[1]
				+ image.at<Vec3b>(row - 1, col)[1] + image.at<Vec3b>(row - 1, col - 1)[1] + image.at<Vec3b>(row - 1, col + 1)[1]
				+ image.at<Vec3b>(row + 1, col)[1] + image.at<Vec3b>(row + 1, col - 1)[1] + image.at<Vec3b>(row + 1, col + 1)[1];
			
			int sum_r = image.at<Vec3b>(row, col)[2] + image.at<Vec3b>(row, col - 1)[2] + image.at<Vec3b>(row, col + 1)[2]
				+ image.at<Vec3b>(row - 1, col)[2] + image.at<Vec3b>(row - 1, col - 1)[2] + image.at<Vec3b>(row - 1, col + 1)[2]
				+ image.at<Vec3b>(row + 1, col)[2] + image.at<Vec3b>(row + 1, col - 1)[2] + image.at<Vec3b>(row + 1, col + 1)[2];
			result.at<Vec3b>(row, col) = Vec3b(sum_b / 9, sum_g / 9, sum_r / 9);
		}
	}
	imshow("�����ʾ", result);
}

void ConvolutionDemo::gaussian_blur_demo(Mat &image) {
	Mat result_size10;
	Mat result_sigma15;
	GaussianBlur(image, result_size10, Size(11, 11), 0, 0, BORDER_DEFAULT);
	GaussianBlur(image, result_sigma15, Size(0, 0), 15, 0, BORDER_DEFAULT);
	imshow("��˹ģ��-10x10", result_size10);
	imshow("��˹ģ��-sigma15", result_sigma15);
}

void ConvolutionDemo::blur_demo(Mat &image) {
	Mat result7x7;
	Mat result15x15;
	blur(image, result7x7, Size(7, 7), Point(-1, -1), BORDER_DEFAULT);
	blur(image, result15x15, Size(15, 15), Point(-1, -1), BORDER_DEFAULT);
	imshow("��ֵģ��-7x7", result7x7);
	imshow("��ֵģ��-15x15", result15x15);
}

void ConvolutionDemo::gradient_demo(Mat &image) {
	Mat gradx, grady;

	// sobel
	Sobel(image, gradx, CV_32F, 1, 0);
	Sobel(image, grady, CV_32F, 0, 1);

	// ��һ����0~1֮��
	normalize(gradx, gradx, 0, 1.0, NORM_MINMAX);
	normalize(grady, grady, 0, 1.0, NORM_MINMAX);

	imshow("�ݶ�-X����", gradx);
	imshow("�ݶ�-Y����", grady);
	waitKey(0);

	// Scharr �ݶ�
	Scharr(image, gradx, CV_32F, 1, 0);
	Scharr(image, grady, CV_32F, 0, 1);

	// ��һ����0~1֮��
	normalize(gradx, gradx, 0, 1.0, NORM_MINMAX);
	normalize(grady, grady, 0, 1.0, NORM_MINMAX);

	imshow("�ݶ�-X����", gradx);
	imshow("�ݶ�-Y����", grady);
}

void ConvolutionDemo::edge_demo(Mat &image) {
	Mat edge;
	int low_T = 150;
	Canny(image, edge, low_T, low_T*2, 3, false);
	imshow("��Ե", edge);
	Mat color_edge;
	bitwise_and(image, image, color_edge, edge);
	imshow("��ɫ��Ե", color_edge);
}

void ConvolutionDemo::epf_demo(Mat &image) {
	Mat denoise_img, cartoon;
	bilateralFilter(image, denoise_img, 7, 80, 10);
	bilateralFilter(image, cartoon, 0, 150, 10);
	imshow("ȥ��Ч��", denoise_img);
	imshow("��ͨЧ��", cartoon);
}

void ConvolutionDemo::denoise_demo(Mat &image) {
	RNG rng(12345);
	int h = image.rows;
	int w = image.cols;
	int nums = 10000;
	Mat jynoise_img = image.clone();
	for (int i = 0; i < nums; i++) {
		int x = rng.uniform(0, w);
		int y = rng.uniform(0, h);
		if (i % 2 == 1) {
			jynoise_img.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
		}
		else {
			jynoise_img.at<Vec3b>(y, x) = Vec3b(0, 0, 0);
		}
	}
	imshow("��������", jynoise_img);

	Mat noise = Mat::zeros(image.size(), image.type());
	randn(noise, (15, 15, 15), (30, 30, 30));
	Mat dst;
	add(image, noise, dst);
	imshow("��˹����", dst);
	imwrite("D:/noisebee.png", dst);

	Mat median_denoise, mean_denoise;
	medianBlur(jynoise_img, median_denoise, 5);
	blur(jynoise_img, mean_denoise, Size(5, 5));
	imshow("��ֵȥ��-5x5", median_denoise);
	imshow("��ֵȥ��-5x5", mean_denoise);
}

void ConvolutionDemo::sharpen_demo(Mat &image) {
	Mat lap_img, sharpen_img;
	Laplacian(image, lap_img, CV_32F, 3, 1.0, 0.0, 4);
	normalize(lap_img, lap_img, 0, 1.0, NORM_MINMAX);
	imshow("������˹", lap_img);
	
	Mat k = (Mat_<float>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(image, sharpen_img, -1, k, Point(-1, -1), 0, BORDER_DEFAULT);
	imshow("��", sharpen_img);

	//  ����ͼ�����
	Mat gray;
	int h = image.rows;
	int w = image.cols;
	float sum = 0;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	for (int row = 1; row < h - 1; row++) {
		for (int col = 1; col < w - 1; col++) {
			int dx = gray.at<uchar>(row, col) * 2 - gray.at<uchar>(row, col + 1) - gray.at<uchar>(row, col - 1);
			int dy = gray.at<uchar>(row, col) * 2 - gray.at<uchar>(row + 1, col) - gray.at<uchar>(row - 1, col);
			sum += (abs(dx) + abs(dy));
		}
	}
	printf("Lapalcian ML sum: %.2f", sum);
}

int main(int argc, char** argv) {
	Mat image = imread(rootdir + "bee.png");
	imshow("����ͼ��", image);
	ConvolutionDemo conv_ops;
	conv_ops.denoise_demo(image);

	waitKey(0);
	return 0;
}