#include <opencv2/opencv.hpp>
#include <histogramdemo.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";
void HistogramDemo::displayHist(Mat &image) {
	
	int bins = 32;
	int histSize[] = {bins};
	float rgb_ranges[] = { 0, 256 };
	const float* ranges[] = { rgb_ranges };
	int channels[] = { 0 };
	int cn = image.channels();
	Mat histImage = Mat::zeros(Size(800, 500), CV_8UC3);
	int padding = 50;
	int hist_w = histImage.cols - 2 * padding;
	int hist_h = histImage.rows - 2 * padding;
	int bin_w = cvRound((double)hist_w / bins);

	if (cn == 1) {
		Mat hist;
		calcHist(&image, 1, channels, Mat(), hist, 1, histSize, ranges, true, false);

		// 绘制直方图曲线
		normalize(hist, hist, 0, hist_h, NORM_MINMAX, -1, Mat());
		int base_h = hist_h + padding;
		for (int i = 1; i < bins; i++) {
			line(histImage, Point(bin_w*(i - 1) + padding, base_h - cvRound(hist.at<float>(i - 1))),
				Point(bin_w*(i)+padding, base_h - cvRound(hist.at<float>(i))), Scalar(255, 255, 255), 2, 8, 0);
		}
	}
	if (cn == 3) {
		std::vector<Mat> mv;
		split(image, mv);
		Mat b_hist, g_hist, r_hist;
		calcHist(&mv[0], 1, channels, Mat(), b_hist, 1, histSize, ranges, true, false);
		calcHist(&mv[1], 1, channels, Mat(), g_hist, 1, histSize, ranges, true, false);
		calcHist(&mv[2], 1, channels, Mat(), r_hist, 1, histSize, ranges, true, false);

		// 归一化直方图数据
		normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
		normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
		normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());

		// 绘制直方图曲线
		int base_h = hist_h + padding;
		for (int i = 1; i < bins; i++) {
			line(histImage, Point(bin_w*(i - 1) + padding, base_h - cvRound(b_hist.at<float>(i - 1))),
				Point(bin_w*(i)+padding, base_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, 8, 0);

			line(histImage, Point(bin_w*(i - 1) + padding, base_h - cvRound(g_hist.at<float>(i - 1))),
				Point(bin_w*(i)+padding, base_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, 8, 0);

			line(histImage, Point(bin_w*(i - 1) + padding, base_h - cvRound(r_hist.at<float>(i - 1))),
				Point(bin_w*(i)+padding, base_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	// 显示直方图
	cv::namedWindow("Histogram Demo", WINDOW_AUTOSIZE);
	cv::imshow("Histogram Demo", histImage);
	waitKey(0);

	// 2D 直方图
	Mat hsv, hs_hist;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	int hbins = 30, sbins = 32;
	int hist_bins[] = { hbins, sbins };
	float h_range[] = { 0, 180 };
	float s_range[] = { 0, 256 };
	const float* hs_ranges[] = { h_range, s_range };
	int hs_channels[] = { 0, 1 };
	calcHist(&hsv, 1, hs_channels, Mat(), hs_hist, 2, hist_bins, hs_ranges, true, false);
	double maxVal = 0;
	minMaxLoc(hs_hist, 0, &maxVal, 0, 0);
	int scale = 10;
	Mat hist2d_image = Mat::zeros(sbins*scale, hbins * scale, CV_8UC3);
	for (int h = 0; h < hbins; h++) {
		for (int s = 0; s < sbins; s++)
		{
			float binVal = hs_hist.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / maxVal);
			rectangle(hist2d_image, Point(h*scale, s*scale),
				Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				Scalar::all(intensity),
				-1);
		}
	}
	imshow("H-S Histogram", hist2d_image);
	imwrite("D:/hist_2d.png", hist2d_image);
}

void HistogramDemo::backProjectionHistogram(Mat &image, Mat &tpl) {
	Mat model_hsv, image_hsv;
	cvtColor(tpl, model_hsv, COLOR_BGR2HSV);
	cvtColor(image, image_hsv, COLOR_BGR2HSV);

	// 定义直方图参数与属性
	int h_bins = 32; int s_bins = 32;
	int histSize[] = { h_bins, s_bins };
	// hue varies from 0 to 179, saturation from 0 to 255
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat roiHist;
	calcHist(&model_hsv, 1, channels, Mat(), roiHist, 2, histSize, ranges, true, false);
	normalize(roiHist, roiHist, 0, 255, NORM_MINMAX, -1, Mat());
	MatND backproj;
	calcBackProject(&image_hsv, 1, channels, roiHist, backproj, ranges, 1.0);
	imshow("BackProj", backproj);
}

void HistogramDemo::cmpHist(Mat &img1, Mat &img2) {
	imshow("input1", img1);
	imshow("input2", img2);

	Mat hsv1, hsv2;
	cvtColor(img1, hsv1, COLOR_BGR2HSV);
	cvtColor(img2, hsv2, COLOR_BGR2HSV);

	int h_bins = 30; int s_bins = 32;
	int histSize[] = { h_bins, s_bins };
	float h_ranges[] = { 0, 180 };
	float s_ranges[] = { 0, 256 };
	const float* ranges[] = { h_ranges, s_ranges };
	int channels[] = { 0, 1 };
	Mat hist1, hist2, hist3, hist4;
	calcHist(&hsv1, 1, channels, Mat(), hist1, 2, histSize, ranges, true, false);
	calcHist(&hsv2, 1, channels, Mat(), hist2, 2, histSize, ranges, true, false);

	normalize(hist1, hist1, 0, 1, NORM_MINMAX, -1, Mat());
	normalize(hist2, hist2, 0, 1, NORM_MINMAX, -1, Mat());
	string methods[] = { "HISTCMP_CORREL" , "HISTCMP_CHISQR" ,
		"HISTCMP_INTERSECT" , "HISTCMP_BHATTACHARYYA" };
	for (int i = 0; i < 4; i++)
	{
		int compare_method = i;
		double src1_src2 = compareHist(hist1, hist2, compare_method);
		printf(" Method [%s]  : src1_src2 : %f  \n", methods[i].c_str(), src1_src2);
	}

}

void HistogramDemo::global_eq(Mat &image) {
	int ch = image.channels();
	if (ch == 1) {
		Mat dst;
		equalizeHist(image, dst);
		imshow("equalizeHist-gray", dst);
	}
	if (ch == 3) {
		Mat hsv, dst;
		std::vector<Mat> mv;
		cvtColor(image, hsv, COLOR_BGR2HSV);
		split(hsv, mv);
		equalizeHist(mv[2], mv[2]);
		merge(mv, dst);
		cvtColor(dst, dst, COLOR_HSV2BGR);
		imshow("equalizeHist-Color", dst);
	}
}

void HistogramDemo::local_eq(Mat &image) {
	int ch = image.channels();
	auto clahe = createCLAHE(2.0, Size(8, 8));
	if (ch == 1) {
		Mat dst;
		clahe->apply(image, dst);
		imshow("clahe-gray", dst);
	}
	if (ch == 3) {
		Mat hsv, dst;
		std::vector<Mat> mv;
		cvtColor(image, hsv, COLOR_BGR2HSV);
		split(hsv, mv);
		clahe->apply(mv[2], mv[2]);
		merge(mv, dst);
		cvtColor(dst, dst, COLOR_HSV2BGR);
		imshow("equalizeHist-Color", dst);
	}
	
}

void HistogramDemo::stasticInfo(Mat &image) {
	Scalar m_bgr = mean(image);
	std::cout << "mean : " << m_bgr << std::endl;
	// Scalar m, std;
	Mat m, std;
	meanStdDev(image, m, std);
	std::cout << "meanStdDev.mean : "<< m << std::endl;
	std::cout << "meanStdDev.dev : " << std << std::endl;
}

int main(int argc, char** argv) {
	//Mat src = imread(rootdir + "lena.jpg");
	//Mat src2 = imread(rootdir + "ela_original.jpg");
	Mat image = imread(rootdir + "fruits.jpg");
	// Mat tpl = imread(rootdir + "tpl.png");
	imshow("input", image);
	HistogramDemo hd;
	// hd.displayHist(src);
	hd.global_eq(image);
	waitKey(0);
	return 0;
}