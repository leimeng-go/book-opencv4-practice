#include <morphology_demo.h>
using namespace cv;
using namespace std;

string rootdir = "D:/opencv-4.8.0/opencv/book_images/";

void MorphologyOpDemo::dilate_erode_demo(Mat &image) {
	Mat dst1, dst2;
	Mat se = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	dilate(image, dst1, se);
	erode(image, dst2, se);
	imshow("����", dst1);
	imshow("��ʴ", dst2);
}

void MorphologyOpDemo::open_close_demo(Mat &image) {
	Mat se = getStructuringElement(MORPH_RECT, Size(9, 9), Point(-1, -1));
	
	// open
	Mat result1;
	erode(image, result1, se);
	dilate(result1, result1, se);
	imshow("������", result1);

	// close
	Mat result2;
	dilate(image, result2, se);
	erode(result2, result2, se);
	imshow("�ղ���", result2);

	Mat result;
	morphologyEx(image, result, MORPH_OPEN, se);
	imshow("MORPH_OPEN", result);
	morphologyEx(image, result, MORPH_CLOSE, se);
	imshow("MORPH_CLOSE", result);
}

void MorphologyOpDemo::gradient_demo(Mat &image) {
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat basic_grad, ex_grad, in_grad;
	Mat di, er;
	dilate(image, di, se);
	erode(image, er, se);
	
	// �����ݶ�
	morphologyEx(image, basic_grad, MORPH_GRADIENT, se);

	// ���ݶ�
	subtract(di, image, ex_grad);

	// ���ݶ�
	subtract(image, er, in_grad);
	// display
	imshow("�����ݶ�", basic_grad);
	imshow("���ݶ�", ex_grad);
	imshow("���ݶ�", in_grad);
}

void MorphologyOpDemo::gradient_edges(Mat &image) {
	Mat se = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Mat gray, edges;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	Mat basic_grad;
	morphologyEx(gray, basic_grad, MORPH_GRADIENT, se);
	threshold(basic_grad, edges, 0, 255, THRESH_BINARY | THRESH_OTSU);
	imshow("��Ե���", edges);
}

void MorphologyOpDemo::hats_demo(Mat &image) {
	Mat se = getStructuringElement(MORPH_RECT, Size(7, 7), Point(-1, -1));
	Mat binary;
	threshold(image, binary, 127, 255, THRESH_BINARY);
	Mat tophat, blackhat;
	morphologyEx(binary, tophat, MORPH_TOPHAT, se);
	morphologyEx(binary, blackhat, MORPH_BLACKHAT, se);
	imshow("��ñ", tophat);
	imshow("��ñ", blackhat);
}

void MorphologyOpDemo::hitandmiss_demo(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
	Mat se1 = (Mat_<int>(3, 3) << 1, 0, 0, 0, -1, 0, 0, 0, 0);
	Mat se2 = (Mat_<int>(3, 3) << 0, 0, 0, 0, -1, 0, 0, 0, 1);
	Mat h1, h2, result;
	morphologyEx(binary, h1, MORPH_HITMISS, se1);
	morphologyEx(binary, h2, MORPH_HITMISS, se2);
	add(h1, h2, result);
	imshow("���л�����", result);
}

void MorphologyOpDemo::hvlines_demo(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	Mat h_line = getStructuringElement(MORPH_RECT, Size(25, 1), Point(-1, -1));
	Mat v_line = getStructuringElement(MORPH_RECT, Size(1, 25), Point(-1, -1));
	Mat hResult, vResult;
	morphologyEx(binary, hResult, MORPH_OPEN, h_line);
	morphologyEx(binary, vResult, MORPH_OPEN, v_line);
	imshow("ˮƽ��", hResult);
	imshow("��ֱ��", vResult);
}

void MorphologyOpDemo::cross_demo(Mat &image) {
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	Mat cross_se = getStructuringElement(MORPH_CROSS, Size(11, 11), Point(-1, -1));
	Mat result;
	morphologyEx(binary, result, MORPH_OPEN, cross_se);
	imshow("ʮ�ֽ���", result);
}

void MorphologyOpDemo::distance_demo(Mat &image) {
	Mat hsv, mask;
 	cvtColor(image, hsv, COLOR_BGR2HSV);
	inRange(hsv, Scalar(150, 200, 200), Scalar(180, 255, 255), mask);
	Mat dst;
	distanceTransform(mask, dst, DIST_L2, 3, CV_32F);
	normalize(dst, dst, 0, 1, NORM_MINMAX);
	
	imshow("����任", dst);
}

void MorphologyOpDemo::wateshed_demo(Mat &image) {
	// ��ֵ��
	Mat gray, binary;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	threshold(gray, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

	// ������
	Mat opening, dist;
	Mat se = getStructuringElement(MORPH_RECT, Size(5, 5), Point(-1, -1));
	morphologyEx(binary, opening, MORPH_OPEN, se);

	// ����
	Mat bg;
	dilate(binary, bg, se);

	// ����任
	distanceTransform(opening, dist, DIST_L2, 3, CV_32F);
	normalize(dist, dist, 0, 1, NORM_MINMAX);

	// ����markers
	Mat objects, markers;
	threshold(dist, objects, 0.7, 1.0, THRESH_BINARY);
	objects = objects*255.0;
	objects.convertTo(objects, CV_8U);
	int num = connectedComponents(objects, markers, 8, 4);

	// ����Ϊ1, unknown = 0, ����label����1
	markers = markers + 1;
	Mat unknown;
	subtract(bg, objects, unknown);
	for (int row = 0; row < unknown.rows; row++) {
		for (int col = 0; col < unknown.cols; col++) {
			int b = unknown.at<uchar>(row, col);
			if (b > 0) {
				markers.at<int>(row, col) = 0;
			}
		}
	}
	imshow("����任", objects);

	// ��ˮ��任
	watershed(image, markers);

	// ������ɫ���ұ�
	RNG rng(12345);
	Vec3b background_color(255, 255, 255);
	std::vector<Vec3b> colors_table;
	for (int i = 1; i < num; i++) {
		int b = rng.uniform(0, 255);
		int g = rng.uniform(0, 255);
		int r = rng.uniform(0, 255);
		colors_table.push_back(Vec3b(b, g, r));
	}

	// ����
	for (int row = 0; row < image.rows; row++) {
		for (int col = 0; col < image.cols; col++) {
			int b = bg.at<uchar>(row, col);
			int c = markers.at<int>(row, col);
			if (b > 0) {
				image.at<Vec3b>(row, col) = colors_table[c+1];
			}
			else {
				image.at<Vec3b>(row, col) = background_color;
			}
		}
	}
	imshow("��ˮ��任", image);
}


int main(int argc, char** argv) {
	// Mat image = imread(rootdir + "water_coins.jpg");
	Mat image = imread(rootdir + "bee.png");
	imshow("����ͼ��", image);
	MorphologyOpDemo mo;
	mo.hitandmiss_demo(image);
	waitKey(0);
	return 0;
}