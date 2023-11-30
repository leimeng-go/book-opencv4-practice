#include "feature_demo.h"
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/images/";
int level = 3;
Mat smallestLevel;
Mat blend(Mat &a, Mat &b, Mat &m) {
	int width = a.cols;
	int height = a.rows;
	Mat dst = Mat::zeros(a.size(), a.type());
	Vec3b rgb1;
	Vec3b rgb2;
	int r1 = 0, g1 = 0, b1 = 0;
	int r2 = 0, g2 = 0, b2 = 0;
	int red = 0, green = 0, blue = 0;
	int w = 0;
	float w1 = 0, w2 = 0;
	for (int row = 0; row<height; row++) {
		for (int col = 0; col<width; col++) {
			rgb1 = a.at<Vec3b>(row, col);
			rgb2 = b.at<Vec3b>(row, col);
			w = m.at<uchar>(row, col);
			w2 = w / 255.0f;
			w1 = 1.0f - w2;

			b1 = rgb1[0] & 0xff;
			g1 = rgb1[1] & 0xff;
			r1 = rgb1[2] & 0xff;

			b2 = rgb2[0] & 0xff;
			g2 = rgb2[1] & 0xff;
			r2 = rgb2[2] & 0xff;

			red = (int)(r1*w1 + r2*w2);
			green = (int)(g1*w1 + g2*w2);
			blue = (int)(b1*w1 + b2*w2);

			// output
			dst.at<Vec3b>(row, col)[0] = blue;
			dst.at<Vec3b>(row, col)[1] = green;
			dst.at<Vec3b>(row, col)[2] = red;
		}
	}
	return dst;
}

vector<Mat> buildGaussianPyramid(Mat &image) {
	vector<Mat> pyramid;
	Mat copy = image.clone();
	pyramid.push_back(image.clone());
	Mat dst;
	for (int i = 0; i<level; i++) {
		pyrDown(copy, dst, Size(copy.cols / 2, copy.rows / 2));
		dst.copyTo(copy);
		pyramid.push_back(dst.clone());
	}
	smallestLevel = dst;
	return pyramid;
}

vector<Mat> buildLapacianPyramid(Mat &image) {
	vector<Mat> lp;
	Mat temp;
	Mat copy = image.clone();
	Mat dst;
	for (int i = 0; i<level; i++) {
		pyrDown(copy, dst, Size(copy.cols / 2, copy.rows / 2));
		pyrUp(dst, temp, copy.size());
		Mat lapaian;
		subtract(copy, temp, lapaian);
		lp.push_back(lapaian);
		copy = dst.clone();
	}
	smallestLevel = dst;
	return lp;
}
void FeatureVectorOps::pyramid_demo(Mat &image) {
	vector<Mat> reduce_images;
	Mat temp = image.clone();
	for (int i = 0; i < level; i++) {
		Mat dst;
		pyrDown(temp, dst);
		// imshow(format("reduce:%d", (i + 1)), dst);
		dst.copyTo(temp);
		reduce_images.push_back(dst);
	}

	for (int i = level - 1; i >= 0; i--) {
		Mat expand;
		Mat lpls;
		if (i - 1 < 0) {
			pyrUp(reduce_images[i], expand, image.size());
			subtract(image, expand, lpls);
		}
		else {
			pyrUp(reduce_images[i], expand, reduce_images[i - 1].size());
			subtract(reduce_images[i - 1], expand, lpls);
		}
		imshow(format("拉普拉斯金字塔:%d", (i + 1)), lpls);
	}
}

void FeatureVectorOps::pyramid_blend_demo(Mat &apple, Mat &orange) {
	Mat mc = imread("D:/images/mask.png");
	if (apple.empty() || orange.empty()) {
		return;
	}
	imshow("苹果图像", apple);
	imshow("橘子图像", orange);

	vector<Mat> la = buildLapacianPyramid(apple);
	Mat leftsmallestLevel;
	smallestLevel.copyTo(leftsmallestLevel);

	vector<Mat> lb = buildLapacianPyramid(orange);
	Mat rightsmallestLevel;
	smallestLevel.copyTo(rightsmallestLevel);

	Mat mask;
	cvtColor(mc, mask, COLOR_BGR2GRAY);

	vector<Mat> maskPyramid = buildGaussianPyramid(mask);
	Mat samllmask;
	smallestLevel.copyTo(samllmask);

	Mat currentImage = blend(leftsmallestLevel, rightsmallestLevel, samllmask);
	imwrite("D:/samll.png", currentImage);
	// 重建拉普拉斯金字塔
	vector<Mat> ls;
	for (int i = 0; i<level; i++) {
		Mat a = la[i];
		Mat b = lb[i];
		Mat m = maskPyramid[i];
		ls.push_back(blend(a, b, m));
	}

	// 重建原图
	Mat temp;
	for (int i = level - 1; i >= 0; i--) {
		pyrUp(currentImage, temp, ls[i].size());
		add(temp, ls[i], currentImage);
	}
	imshow("高斯金子图像融合重建-图像", currentImage);
}

void FeatureVectorOps::harris_demo(Mat &image) {
	RNG rng(12345);

	// parameters
	int blockSize = 2;
	int apertureSize = 3;
	double k = 0.04;

	// 角点检测
	Mat gray, dst;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	cornerHarris(gray, dst, blockSize, apertureSize, k);

	// 归一化
	Mat dst_norm = Mat::zeros(dst.size(), dst.type());
	normalize(dst, dst_norm, 0, 255, NORM_MINMAX);
	convertScaleAbs(dst_norm, dst_norm);

	// 绘制角点
	for (int row = 0; row < dst_norm.rows; row++) {
		for (int col = 0; col < dst_norm.cols; col++) {
			int rsp = dst_norm.at<uchar>(row, col);
			if (rsp > 130) {
				int b = rng.uniform(0, 256);
				int g = rng.uniform(0, 256);
				int r = rng.uniform(0, 256);
				circle(image, Point(row, col), 5, Scalar(b, g, r), 2);
			}
		}
	}
	imshow("harris角点检测", image);
}

void FeatureVectorOps::shi_tomas_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	int maxCorners = 400;
	double qualityLevel = 0.01;
	std::vector<Point> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, 5, Mat(), 3, false, 0.04);

	// 绘制
	RNG rng(12345);
	for (size_t t = 0; t < corners.size(); t++) {
		Point pt = corners[t];
		int b = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int r = rng.uniform(0, 256);
		circle(image, pt, 3, Scalar(b, g, r), 2, 8, 0);
	}
	imshow("shi-tomas-corner-detector", image);
}

void FeatureVectorOps::corners_sub_pixels_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	int maxCorners = 400;
	double qualityLevel = 0.01;
	std::vector<Point2f> corners;
	goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, 5, Mat(), 3, false, 0.04);

	Size winSize = Size(5, 5);
	Size zeroZone = Size(-1, -1);
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.001);
	cornerSubPix(gray, corners, winSize, zeroZone, criteria);
	for (size_t t = 0; t < corners.size(); t++) {
		printf("refined Corner: %d, x:%.2f, y:%.2f\n", t, corners[t].x, corners[t].y);
	}
}

void FeatureVectorOps::hog_feature_demo(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	HOGDescriptor hogDetector;
	std::vector<float> hog_descriptors;
	hogDetector.compute(gray, hog_descriptors, Size(8, 8), Size(0, 0));
	std::cout << hog_descriptors.size() << std::endl;
	for (size_t t = 0; t < hog_descriptors.size(); t++) {
		std::cout << hog_descriptors[t] << std::endl;
	}
}

void FeatureVectorOps::hog_detect_demo(Mat &image) {
	HOGDescriptor *hog = new HOGDescriptor();
	hog->setSVMDetector(hog->getDefaultPeopleDetector());
	vector<Rect> objects;
	hog->detectMultiScale(image, objects, 0.0, Size(4, 4), Size(8, 8), 1.25);
	for (int i = 0; i < objects.size(); i++) {
		rectangle(image, objects[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("HOG行人检测", image);
}

void FeatureVectorOps::orb_detect_demo(Mat &image) {
	auto orb_detector = ORB::create();
	std::vector<KeyPoint> kpts;
	orb_detector->detect(image, kpts, Mat());
	Mat outImage;
	drawKeypoints(image, kpts, outImage, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	imshow("ORB特征点提取", outImage);
}

void FeatureVectorOps::orb_match_demo(Mat &box, Mat &box_in_scene) {
	// ORB特征提取
	auto orb_detector = ORB::create();
	std::vector<KeyPoint> box_kpts;
	std::vector<KeyPoint> scene_kpts;
	Mat box_descriptors, scene_descriptors;
	orb_detector->detectAndCompute(box, Mat(), box_kpts, box_descriptors);
	orb_detector->detectAndCompute(box_in_scene, Mat(), scene_kpts, scene_descriptors);

	// 暴力匹配
	auto bfMatcher = BFMatcher::create(NORM_HAMMING, false);
	std::vector<DMatch> matches;
	bfMatcher->match(box_descriptors, scene_descriptors, matches);
	Mat img_orb_matches;
	drawMatches(box, box_kpts, box_in_scene, scene_kpts, matches, img_orb_matches);
	imshow("ORB暴力匹配演示", img_orb_matches);

	// FLANN匹配
	auto flannMatcher = FlannBasedMatcher(new flann::LshIndexParams(6, 12, 2));
	flannMatcher.match(box_descriptors, scene_descriptors, matches);
	Mat img_flann_matches;
	drawMatches(box, box_kpts, box_in_scene, scene_kpts, matches, img_flann_matches);
	namedWindow("FLANN匹配演示", WINDOW_FREERATIO);
	cv::namedWindow("FLANN匹配演示", cv::WINDOW_NORMAL);
	imshow("FLANN匹配演示", img_flann_matches);
}

void FeatureVectorOps::find_known_object(Mat &book, Mat &book_on_desk) {
	// ORB特征提取
	auto orb_detector = ORB::create();
	std::vector<KeyPoint> box_kpts;
	std::vector<KeyPoint> scene_kpts;
	Mat box_descriptors, scene_descriptors;
	orb_detector->detectAndCompute(book, Mat(), box_kpts, box_descriptors);
	orb_detector->detectAndCompute(book_on_desk, Mat(), scene_kpts, scene_descriptors);

	// 暴力匹配
	auto bfMatcher = BFMatcher::create(NORM_HAMMING, false);
	std::vector<DMatch> matches;
	bfMatcher->match(box_descriptors, scene_descriptors, matches);
	
	// 好的匹配
	std::sort(matches.begin(), matches.end());
	const int numGoodMatches = matches.size() * 0.15;
	matches.erase(matches.begin() + numGoodMatches, matches.end());
	Mat img_bf_matches;
	drawMatches(book, box_kpts, book_on_desk, scene_kpts, matches, img_bf_matches);
	imshow("ORB暴力匹配演示", img_bf_matches);

	// 单应性求H
	std::vector<Point2f> obj_pts;
	std::vector<Point2f> scene_pts;
	for (size_t i = 0; i < matches.size(); i++)
	{
		//-- Get the keypoints from the good matches
		obj_pts.push_back(box_kpts[matches[i].queryIdx].pt);
		scene_pts.push_back(scene_kpts[matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj_pts, scene_pts, RANSAC);
	std::cout << "RANSAC estimation parameters: \n" << H << std::endl;
	std::cout << std::endl;
	H = findHomography(obj_pts, scene_pts, RHO);
	std::cout << "RHO estimation parameters: \n" << H << std::endl;
	std::cout << std::endl;
	H = findHomography(obj_pts, scene_pts, LMEDS);
	std::cout << "LMEDS estimation parameters: \n" << H << std::endl;

	// 变换矩阵得到目标点
	std::vector<Point2f> obj_corners(4);
	obj_corners[0] = Point(0, 0); obj_corners[1] = Point(book.cols, 0);
	obj_corners[2] = Point(book.cols, book.rows); obj_corners[3] = Point(0, book.rows);
	std::vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);

	// 绘制结果
	Mat dst;
	line(img_bf_matches, scene_corners[0] + Point2f(book.cols, 0), scene_corners[1] + Point2f(book.cols, 0), Scalar(0, 255, 0), 4);
	line(img_bf_matches, scene_corners[1] + Point2f(book.cols, 0), scene_corners[2] + Point2f(book.cols, 0), Scalar(0, 255, 0), 4);
	line(img_bf_matches, scene_corners[2] + Point2f(book.cols, 0), scene_corners[3] + Point2f(book.cols, 0), Scalar(0, 255, 0), 4);
	line(img_bf_matches, scene_corners[3] + Point2f(book.cols, 0), scene_corners[0] + Point2f(book.cols, 0), Scalar(0, 255, 0), 4);

	//-- Show detected matches
	namedWindow("基于特征的对象检测", cv::WINDOW_NORMAL);
	imshow("基于特征的对象检测", img_bf_matches);
}

int main(int argc, char** argv) {
	 Mat apple = imread("D:/images/book.jpg");
	 Mat orange = imread("D:/images/book_on_desk.jpg");
	 FeatureVectorOps fvo;
	 fvo.find_known_object(apple, orange);
	//Mat book = imread(rootdir + "book.jpg");
	//Mat book_in_scene = imread(rootdir + "book_on_desk.png");
	//Mat image = imread(rootdir + "bee.png");
	//imshow("输入图像", image);
	//FeatureVectorOps fvo;
	//fvo.orb_detect_demo(image);
	/*Mat image = imread("D:/111.jpg");
	Mat dst;
	int colormap[]= {
		COLORMAP_AUTUMN,
		COLORMAP_BONE,
		COLORMAP_JET,
		COLORMAP_WINTER,
		COLORMAP_RAINBOW,
		COLORMAP_OCEAN,
		COLORMAP_SUMMER,
		COLORMAP_SPRING,
		COLORMAP_COOL,
		COLORMAP_PINK,
		COLORMAP_HOT,
		COLORMAP_PARULA,
		COLORMAP_MAGMA,
		COLORMAP_INFERNO,
		COLORMAP_PLASMA,
		COLORMAP_VIRIDIS,
		COLORMAP_CIVIDIS,
		COLORMAP_TWILIGHT,
		COLORMAP_TWILIGHT_SHIFTED
	};
	for (int i = 0; i < 19; i++) {
		applyColorMap(image, dst, colormap[i]);
		imwrite(format("D:/%d.jpg", i), dst);
	}

	bilateralFilter(image, dst, 0, 50, 10);
	imwrite("D:/myblur.png", dst);
	*/
	waitKey(0);
	return 0;
}