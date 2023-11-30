#include <defectdetection.h>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";
// please download models from 
// https://github.com/gloomyfish1998/opencv_tutorial
string model_dir = "D:/projects/unet_road.onnx";

void defectDetection::basic_blob_detection(Mat &image) {
	Mat binaryDark, binaryLight, blurMat;
	blur(image, blurMat, Size(3, 3));
	threshold(blurMat, binaryDark, 100, 255, THRESH_BINARY_INV);
	threshold(blurMat, binaryLight, 200, 255, THRESH_BINARY);
	Mat result;
	add(binaryDark, binaryLight, result);
	imshow("Blob斑点分析", result);
}

void defectDetection::basic_flaw_detection(Mat &image) {
	// 二值处理
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	cv::Mat  imagemean, diff, binary;
	blur(gray, imagemean, Size(13, 13));
	subtract(imagemean, gray, diff);
	threshold(diff, binary, 5, 255, THRESH_BINARY_INV);

	// 轮廓发现
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	cv::bitwise_not(binary, binary);
	findContours(binary, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, Point(0, 0));

	// 轮廓+阈值分析
	Mat result = Mat::zeros(gray.size(), CV_8U);
	for (int i = 0; i < contours.size(); i++) {
		Moments moms = moments(Mat(contours[i]));
		double area = moms.m00;
		if (area > 20 && area < 1000) {
			drawContours(result, contours, i, Scalar(255), FILLED, 8, hierarchy, 0, Point());
		}
	}
	namedWindow("简单划痕分析", cv::WINDOW_NORMAL);
	imshow("简单划痕分析", result);
}

void defectDetection::hard_blob_detection(Mat &image) {
	Mat blurMat, binary;

	//图像尺寸的修正目的在于计算过程中可以优化，加快计算
	int h = cv::getOptimalDFTSize(image.rows);
	int w = cv::getOptimalDFTSize(image.cols);
	Mat padded;

	//扩充成最优尺寸, 常量填充，为黑色
	copyMakeBorder(image, padded, 0, h - image.rows, 0, w - image.cols, BORDER_CONSTANT, Scalar::all(0));

	//将原图扩充为实部和虚部
	//实部为img，虚部填充0
	Mat plane[] = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) }; 
	Mat complexImg;
	merge(plane, 2, complexImg);

	//离散傅里叶变换，到频域
	dft(complexImg, complexImg);

	//构建一个中频滤波器
	Mat filter(complexImg.size(), CV_32FC1);
	Point filterCenter(filter.cols / 2, filter.rows / 2);
	double W = 10;//带宽，决定增强中频的范围
	double D0 = sqrt((pow(filterCenter.x, 2) + pow(filterCenter.y, 2)));
	for (int r = 0; r < filter.rows; r++)
	{
		float* data = filter.ptr<float>(r);
		for (int c = 0; c < filter.cols; c++)
		{
			double Duv = sqrt(pow(filterCenter.x - c, 2) + pow((filterCenter.y - r), 2));
			if (abs(Duv - D0 / 2)< W) {
				data[c] = 1;
			} 
			else {
				data[c] = 0.5;
			}
		}
	}
	//对四个频谱位置进行交换
	cv::Mat temp = filter.clone();
	temp(cv::Rect(0, 0, temp.cols / 2, temp.rows / 2)).copyTo(filter(cv::Rect(temp.cols / 2, temp.rows / 2, temp.cols / 2, temp.rows / 2)));//左上到右下
	temp(cv::Rect(temp.cols / 2, 0, temp.cols / 2, temp.rows / 2)).copyTo(filter(cv::Rect(0, temp.rows / 2, temp.cols / 2, temp.rows / 2)));//右上到左下
	temp(cv::Rect(0, temp.rows / 2, temp.cols / 2, temp.rows / 2)).copyTo(filter(cv::Rect(temp.cols / 2, 0, temp.cols / 2, temp.rows / 2)));//左下到右上
	temp(cv::Rect(temp.cols / 2, temp.rows / 2, temp.cols / 2, temp.rows / 2)).copyTo(filter(cv::Rect(0, 0, temp.cols / 2, temp.rows / 2)));//右下到左上
	
	//将中频滤波器赛道一个两通道的mat里面
	Mat butterworth_channels[] = { Mat_<float>(filter), Mat::zeros(filter.size(), CV_32F) };
	merge(butterworth_channels, 2, filter);

	//进行频域卷积滤波运算
	mulSpectrums(complexImg, filter, complexImg, 0);

	//反变换回到空域中去
	cv::Mat spatial;
	cv::idft(complexImg, spatial, cv::DFT_SCALE);
	std::vector<cv::Mat> planes;
	cv::split(spatial, planes);
	cv::magnitude(planes[0], planes[1], spatial);

	//归一化
	normalize(spatial, spatial, 0, 255, cv::NORM_MINMAX);
	spatial.convertTo(spatial, CV_8UC1);

	//对提取图像中灰度变化距离的区域
	cv::Mat light, dark;
	cv::threshold(spatial, dark, 10, 255, 1);
	cv::threshold(spatial, light, 230, 255, 0);
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
	cv::dilate(dark, dark, kernel);
	cv::dilate(light, light, kernel);
	cv::bitwise_and(light, dark, light);

	//轮廓查找
	vector<vector<cv::Point>> contours;
	cv::findContours(light, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++) {
		cv::Rect rect = cv::boundingRect(contours[i]);
		cv::rectangle(image, cv::Rect(rect.x - 10, rect.y - 10, 20, 20), 255, 1);
	}
	imshow("基于频域增强的复杂背景缺陷分析", image);
}

void defectDetection::hard_flaw_detection(Mat &image) {
	Mat gray;
	cv::cvtColor(image, gray, COLOR_BGR2GRAY);

	//计算直方图,用频率最高的灰度值作为全图的背景图
	int histsize = 256;
	float range[] = { 0,256 };
	const float*histRanges = { range };
	cv::Mat hist;
	calcHist(&gray, 1, 0, cv::Mat(), hist, 1, &histsize, &histRanges, true, false);
	double maxVal = 0;
	cv::Point maxLoc;
	minMaxLoc(hist, NULL, &maxVal, NULL, &maxLoc);
	cv::Mat BackImg(gray.size(), CV_8UC1, maxLoc.y);

	//将背景图和原图进行差分，增强图像间的差异
	BackImg.convertTo(BackImg, CV_32FC1);
	gray.convertTo(gray, CV_32FC1);
	cv::Mat subImage = 50 + 3 * (gray - BackImg);
	subImage.convertTo(subImage, CV_8UC1);

	//使用中值滤波抑制小斑点和细线
	cv::Mat BlurImg;
	cv::medianBlur(subImage, BlurImg, 15);
	cv::Mat Binary;
	cv::threshold(BlurImg, Binary, 40, 255, cv::THRESH_BINARY_INV);
	imwrite("D:/binary.png", Binary);
	vector<vector<cv::Point>> contours;
	cv::findContours(Binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
	for (int i = 0; i < contours.size(); i++)
	{
		int area = cv::contourArea(contours[i]);
		if (area > 300)
			cv::drawContours(image, contours, i, cv::Scalar(255, 255, 255), 2);
	}
	imwrite("D:/result.png", image);
	cv::namedWindow("复杂背景划痕检测", cv::WINDOW_NORMAL);
	imshow("复杂背景划痕检测", image);

}

void sort_box(vector<Rect> &boxes) {
	int size = boxes.size();
	for (int i = 0; i < size - 1; i++) {
		for (int j = i; j < size; j++) {
			int x = boxes[j].x;
			int y = boxes[j].y;
			if (y < boxes[i].y) {
				Rect temp = boxes[i];
				boxes[i] = boxes[j];
				boxes[j] = temp;
			}
		}
	}
}

void detect_defect(Mat &binary, vector<Rect> rects, vector<Rect> &defect, Mat &tpl) {
	int h = tpl.rows;
	int w = tpl.cols;
	int size = rects.size();
	for (int i = 0; i < size; i++) {
		// 构建diff
		Mat roi = binary(rects[i]);
		resize(roi, roi, tpl.size());
		Mat mask;
		subtract(tpl, roi, mask);
		Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
		morphologyEx(mask, mask, MORPH_OPEN, se);
		threshold(mask, mask, 0, 255, THRESH_BINARY);
		// imshow("mask", mask);
		// waitKey(0);

		// 根据diff查找缺陷，阈值化
		int count = 0;
		for (int row = 0; row < h; row++) {
			for (int col = 0; col < w; col++) {
				int pv = mask.at<uchar>(row, col);
				if (pv == 255) {
					count++;
				}
			}
		}

		// 填充一个像素宽
		int mh = mask.rows + 2;
		int mw = mask.cols + 2;
		Mat m1 = Mat::zeros(Size(mw, mh), mask.type());
		Rect mroi;
		mroi.x = 1;
		mroi.y = 1;
		mroi.height = mask.rows;
		mroi.width = mask.cols;
		mask.copyTo(m1(mroi));

		// 轮廓分析
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(m1, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
		bool find = false;
		for (size_t t = 0; t < contours.size(); t++) {
			Rect rect = boundingRect(contours[t]);
			float ratio = (float)rect.width / ((float)rect.height);
			if (ratio > 4.0 && (rect.y < 5 || (m1.rows - (rect.height + rect.y)) < 10)) {
				continue;
			}
			double area = contourArea(contours[t]);
			if (area > 10) {
				printf("ratio : %.2f, area : %.2f \n", ratio, area);
				find = true;
			}
		}

		if (count > 50 && find) {
			printf("count : %d \n", count);
			defect.push_back(rects[i]);
		}
	}
}

void defectDetection::multiple_defects_detection(Mat &src) {
	Mat tpl = imread(rootdir + "dt.png", IMREAD_GRAYSCALE);

	// 图像二值化
	Mat gray, binary;
	cvtColor(src, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("binary", binary);
	imwrite("D:/binary.png", binary);

	// 定义结构元素
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	morphologyEx(binary, binary, MORPH_OPEN, se);

	// 轮廓发现
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	vector<Rect> rects;
	findContours(binary, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	int height = src.rows;
	for (size_t t = 0; t < contours.size(); t++) {
		Rect rect = boundingRect(contours[t]);
		double area = contourArea(contours[t]);
		if (rect.height >(height / 2)) {
			continue;
		}
		if (area < 150) {
			continue;
		}
		//imshow("roi", binary(rect));
		//waitKey(0);
		rects.push_back(rect);
	}

	// 对每个刀片进行比对检测
	sort_box(rects);
	vector<Rect> defects;
	detect_defect(binary, rects, defects, tpl);

	// 显示检测结果
	for (int i = 0; i < defects.size(); i++) {
		rectangle(src, defects[i], Scalar(0, 0, 255), 2, 8, 0);
		putText(src, "bad", defects[i].tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
	}
	imshow("多个缺陷检测", src);
}

void defectDetection::resnet_surface_detection(Mat &image){
	String defect_labels[] = { "In","Sc","Cr","PS","RS","Pa" };
	dnn::Net net = dnn::readNetFromONNX(model_dir + "surface_defect_resnet18.onnx");
	Mat inputBlob = dnn::blobFromImage(image, 0.00392, Size(200, 200), Scalar(127, 127, 127), false, false);
	inputBlob /= 0.5;

	// 执行图像分类
	Mat prob;
	net.setInput(inputBlob);
	prob = net.forward();

	// 得到最可能分类输出
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f\n", defect_labels[classidx].c_str(), classProb);

	// 显示文本
	putText(image, defect_labels[classidx].c_str(), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("基于分类的缺陷检测", image);
}

void defectDetection::segnet_surface_detection(Mat &image) {
	int h = image.rows;
	int w = image.cols;

	// 输出格式
	int out_h = 320;
	int out_w = 480;
	// 加载模型
	dnn::Net net = dnn::readNetFromONNX(model_dir);

	// 转换为输入格式1x1x320x480 , 0~1 取值范围，浮点数
	Mat inputBlob = dnn::blobFromImage(image, 0.00392, Size(480, 320), Scalar(), false, false);

	// 执行预测
	Mat prob;
	net.setInput(inputBlob);
	cv::Mat preds = net.forward();
	const float* detection = preds.ptr<float>();
	cv::Mat result = cv::Mat::zeros(cv::Size(out_w, out_h), CV_32FC1);

	// 解析输出结果
	for (int row = 0; row < out_h; row++) {
		for (int col = 0; col < out_w; col++) {
			float c1 = detection[row * out_w + col];
			float c2 = detection[out_h*out_w + row * out_w + col];
			if (c1 > c2) {
				result.at<float>(row, col) = 0;
			}
			else {
				result.at<float>(row, col) = 1;
			}
		}
	}
	result = result * 255;
	result.convertTo(result, CV_8U);
	imshow("输入图像", image);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(result, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), -1, 8);
	imshow("OpenCV DNN + UNet道路裂纹检测", image);
}

int main(int argc, char** argv) {
	Mat image = imread(rootdir + "bflaw.jpg");
	defectDetection inspector;
	cv::namedWindow("原图", cv::WINDOW_NORMAL);
	cv::imshow("原图", image);
	inspector.basic_flaw_detection(image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}