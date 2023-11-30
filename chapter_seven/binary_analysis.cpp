#include <binary_analysis.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";

void BinaryAnalysis::find_contours_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	Mat result = Mat::zeros(image.size(), image.type());
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2, 8);
	imshow("轮廓发现", result);
}

void contours_info(Mat &image, vector<vector<Point>> &contours) {
	Mat gray, binary;
	vector<Vec4i> hierarchy;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
}

void BinaryAnalysis::contour_match_demo(Mat &image) {
	Mat src = imread("D:/images/abc.png");
	imshow("input", src);
	Mat src2 = imread("D:/images/a5.png");
	namedWindow("input2", WINDOW_FREERATIO);
	imshow("input2", src2);

	// 轮廓提取
	vector<vector<Point>> contours1;
	vector<vector<Point>> contours2;
	contours_info(src, contours1);
	contours_info(src2, contours2);
	// hu矩计算
	Moments mm2 = moments(contours2[0]);
	Mat hu2;
	HuMoments(mm2, hu2);
	// 轮廓匹配
	for (size_t t = 0; t < contours1.size(); t++) {
		Moments mm = moments(contours1[t]);
		Mat hum;
		HuMoments(mm, hum);
		double dist = matchShapes(hum, hu2, CONTOURS_MATCH_I1, 0);
		printf("contour match distance : %.2f\n", dist);
		if (dist < 1) {
			printf("draw it \n");
			Rect box = boundingRect(contours1[t]);
			rectangle(src, box, Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("match result", src);
}

void BinaryAnalysis::inner_extenerl_circle_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(gray, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (size_t t = 0; t < contours.size(); t++) {
		// 最小外接圆
		Point2f pt;
		float radius;
		minEnclosingCircle(contours[t], pt, radius);
		circle(image, pt, radius, Scalar(255, 0, 0), 2, 8, 0);

		// 点多边形测试
		Mat raw_dist(image.size(), CV_32F);
		for (int i = 0; i < image.rows; i++)
		{
			for (int j = 0; j < image.cols; j++)
			{
				raw_dist.at<float>(i, j) = (float)pointPolygonTest(contours[t], Point2f((float)j, (float)i), true);
			}
		}

		// 获取最大内接圆半径
		double minVal, maxVal;
		Point maxDistPt; // inscribed circle center
		minMaxLoc(raw_dist, &minVal, &maxVal, NULL, &maxDistPt);
		minVal = abs(minVal);
		maxVal = abs(maxVal);
		circle(image, maxDistPt, maxVal, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("最大内接圆与最小外接圆演示", image);
}

void BinaryAnalysis::hough_line_demo(Mat &image) {
	Mat edges;
	Canny(image, edges, 50, 200, 3);
	vector<Vec2f> lines;
	HoughLines(edges, lines, 1, CV_PI / 180, 150, 0, 0);
	Mat result1, result2;
	cvtColor(edges, result1, COLOR_GRAY2BGR);
	result2 = result1.clone();
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(result1, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	imshow("标准霍夫直线检测", result1);

	// 概率霍夫直线检测
	vector<Vec4i> linesP;
	HoughLinesP(edges, linesP, 1, CV_PI / 180, 50, 50, 10);
	for (size_t t = 0; t < linesP.size(); t++) {
		Point p1 = Point(linesP[t][0], linesP[t][1]);
		Point p2 = Point(linesP[t][2], linesP[t][3]);
		line(result2, p1, p2, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("概率霍夫直线检测", result2);
}

void BinaryAnalysis::hough_circle_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(5, 5), 0, 0);
	std::vector<Vec3f> circles;
	HoughCircles(gray, circles, HOUGH_GRADIENT_ALT, 2, 10, 100, 50, 20, 40);
	for (size_t t = 0; t < circles.size(); t++) {
		Vec3f c = circles[t];
		Point center = Point(c[0], c[1]);
		int radius = c[2];
		circle(image, center, radius, Scalar(255, 0, 255), 2, 8, 0); 
		circle(image, center, 3, Scalar(255, 0, 0), 3, 8, 0);
	}
	imshow("霍夫圆检测", image);
}

void BinaryAnalysis::contours_attrs_demo(Mat &image) {
	// 二值化
	Mat edges;
	int t = 80;
	Canny(image, edges, t, t * 2, 3, false);

	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(edges, edges, k);

	// 轮廓发现
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	Mat result = Mat::zeros(image.size(), image.type());
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2, 8);
	imshow("轮廓发现", result);

	Mat mask = Mat::zeros(image.size(), CV_8UC1);
	for (size_t t = 0; t < contours.size(); t++) {
		Rect box = boundingRect(contours[t]);
		RotatedRect rrt = minAreaRect(contours[t]);
		std::vector<Point> hulls;
		convexHull(contours[t], hulls);
		double hull_area = contourArea(hulls);
		double box_area = box.width*box.height;
		double area = contourArea(contours[t]);
		// 计算横纵比
		double aspect_ratio = saturate_cast<double>(rrt.size.width) / saturate_cast<double>(rrt.size.height);
		// 计算延展度
		double extent = area / box_area;
		// 计算实密度
		double solidity = area / hull_area;
		// 生成mask与计算像素均值
		mask.setTo(Scalar(0));
		drawContours(mask, contours, t, Scalar(255), -1);
		Scalar bgra = mean(image, mask);
		putText(image, format("extent:%.2f", extent), box.tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, 8);
		putText(image, format("solidity:%.2f", solidity), Point(box.x, box.y + 14), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, 8);
		putText(image, format("aspect_ratio:%.2f", aspect_ratio), Point(box.x, box.y + 28), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, 8);
		putText(image, format("mean:(%d,%d,%d)", (int)bgra[0], (int)bgra[1], (int)bgra[2]), Point(box.x, box.y + 42), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 0, 255), 1, 8);
	}
	imshow("轮廓分析", image);
}

void BinaryAnalysis::contours_apprv_demo(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	double t = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	for (size_t t = 0; t < contours.size(); t++) {
		std::vector<Point> pts;
		approxPolyDP(contours[t], pts, 10, true);
		for (int i = 0; i < pts.size(); i++) {
			circle(image, pts[i], 3, Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("轮廓逼近", image);
}

void BinaryAnalysis::contours_fitness_demo(Mat &image) {
	// 二值化
	Mat edges;
	int t = 80;
	Canny(image, edges, t, t * 2, 3, false);

	Mat k = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	dilate(edges, edges, k);

	// 轮廓发现
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());

	for (size_t t = 0; t < contours.size(); t++) {
		if (contours[t].size() < 5) {
			continue;
		}
		// 拟合椭圆
		// RotatedRect rrt = fitEllipse(contours[t]);
		// ellipse(image, rrt, Scalar(0, 0, 255), 2, 8);

		// 拟合直线
		Vec4f oneline;
		fitLine(contours[t], oneline, DIST_L1, 0, 0.01, 0.01);
		float vx = oneline[0];
		float vy = oneline[1];
		float x0 = oneline[2];
		float y0 = oneline[3];

		// 直线参数斜率k与截矩b
		float k = vy / vx;
		float b = y0 - k*x0;
		// 寻找轮廓极值点
		int minx = 0, miny = 10000;
		int maxx = 0, maxy = 0;
		for (int i = 0; i < contours[t].size(); i++) {
			Point pt = contours[t][i];
			if (miny > pt.y) {
				miny = pt.y;
			}
			if (maxy < pt.y) {
				maxy = pt.y;
			}
		}
		maxx = (maxy - b) / k;
		minx = (miny - b) / k;
		line(image, Point(maxx, maxy), Point(minx, miny), Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("轮廓拟合-直线拟合", image);
}

void BinaryAnalysis::contours_analysis_demo(Mat &image) {
	// 二值化
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	double t = threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 轮廓发现
	std::vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point());
	Mat result = Mat::zeros(image.size(), image.type());
	drawContours(result, contours, -1, Scalar(0, 0, 255), 2, 8);

	// 轮廓测量
	for (size_t t = 0; t < contours.size(); t++) {
		Rect box = boundingRect(contours[t]);
		
		double area = contourArea(contours[t]);
		double arc = arcLength(contours[t], true);
		putText(result, format("area:%.2f", area), box.tl(), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1, 8);
		putText(result, format("arc:%.2f", arc), Point(box.x, box.y+14), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 1, 8);
	}
	imshow("轮廓测量", result);
}

void BinaryAnalysis::connected_component_demo(Mat &image) {
	Mat gray, binary;
	// 二值化
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat centeroids, labels, stats;

	// 连通组件扫描
	int nums = connectedComponentsWithStats(binary, labels, stats, centeroids, 8, 4);
	cvtColor(binary, binary, COLOR_GRAY2BGR);
	//for (int row = 0; row < labels.rows; row++) {
	//	for (int col = 0; col < labels.cols; col++) {
	//		if (labels.at<int>(row, col) > 0) {
	//			cout << row << ", " << col << " = " << labels.at<int>(row, col)<<std::endl;
	//		}
	//	}
	//}

	// 显示统计信息
	for (int i = 1; i < nums; i++) {
		int x = centeroids.at<double>(i, 0);
		int y = centeroids.at<double>(i, 1);
		int left = stats.at<int>(i, CC_STAT_LEFT);
		int top = stats.at<int>(i, CC_STAT_TOP);
		int width = stats.at<int>(i, CC_STAT_WIDTH);
		int height = stats.at<int>(i, CC_STAT_HEIGHT);
		int area = stats.at<int>(i, CC_STAT_AREA);
		Rect box(left, top, width, height);
		rectangle(binary, box, Scalar(0, 255, 0), 2, 8, 0);
		circle(binary, Point(x, y), 2, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("二值连通组件演示", binary);
}


void BinaryAnalysis::binary_ops(Mat &image) {
	Mat hsv, mask;
	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(20, 43, 46), Scalar(180, 255, 255), mask);
	imshow("inRange", mask);

	Mat edges;
	int t = 80;
	Canny(image, edges, t, t*2, 3, false);
	imshow("边缘检测", edges);

	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	imshow("OTSU二值化", binary);

	adaptiveThreshold(gray, binary, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 25, 10);
	imshow("自适应二值化", binary);
}

void BinaryAnalysis::max_contour_demo(Mat &image) {
	// 二值图像
	Mat mask;
	inRange(image, Scalar(0, 0, 0), Scalar(110, 110, 110), mask);
	bitwise_not(mask, mask);

	// 轮廓发现
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int height = image.rows;
	int width = image.cols;
	int index = -1;
	int max = 0;

	// 最大轮廓寻找
	for (size_t t = 0; t < contours.size(); t++) {
		double area = contourArea(contours[t]);
		if (area > max) {
			max = area;
			index = t;
		}
	}
	Mat result = Mat::zeros(image.size(), image.type());
	Mat pts;
	drawContours(result, contours, index, Scalar(0, 0, 255), 1, 8);

	// 关键点编码提取与绘制
	approxPolyDP(contours[index], pts, 4, true);
	for (int i = 0; i < pts.rows; i++) {
		Vec2i pt = pts.at<Vec2i>(i, 0);
		circle(result, Point(pt[0], pt[1]), 2, Scalar(0, 255, 0), 2, 8, 0);
		circle(result, Point(pt[0], pt[1]), 2, Scalar(0, 255, 0), 2, 8, 0);
	}
	imshow("最大轮廓与关键点编码", result);
}

void BinaryAnalysis::convex_demo(Mat &image) {
	vector<vector<Point>> contours;
	contours_info(image, contours);
	for (size_t t = 0; t < contours.size(); t++) {
		vector<Point> hull;
		convexHull(contours[t], hull);
		bool isHull = isContourConvex(contours[t]);
		printf("test convex of the contours %s \n", isHull ? "Y" : "N");
		int len = hull.size();
		for (int i = 0; i < hull.size(); i++) {
			circle(image, hull[i], 4, Scalar(255, 0, 0), 2, 8, 0);
			line(image, hull[i%len], hull[(i + 1) % len], Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	imshow("凸包检测", image);
}

int main(int argc, char** argv) {
	// Mat image = imread(rootdir + "convex.png");
	Mat image = imread(rootdir + "circleDets.png");
	//cv::bitwise_not(image, image);
	imshow("输入图像", image);
	BinaryAnalysis ca;
	ca.inner_extenerl_circle_demo(image);
	waitKey(0);
	return 0;
}