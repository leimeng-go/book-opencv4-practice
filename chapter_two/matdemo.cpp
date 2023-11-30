#include <matdemo.h>
string rootdir = "D:/opencv-4.5.4/opencv/sources/samples/data/";
void MatDemo::mat_ops() {
	// create Mat - 1
	Mat m1(4, 4, CV_8UC1, Scalar(255));
	std::cout << "m1:\n" << m1 << std::endl;

	// create Mat - 2
	Mat m2(Size(4, 4), CV_8UC3, Scalar(0, 0, 255));
	std::cout << "m2:\n" << m2 << std::endl;

	// create Mat - 3
	Mat m3(Size(4, 4), CV_8UC3, Scalar::all(255));
	std::cout << "m3:\n" << m3 << std::endl;

	// create Matlab风格 - 4
	Mat m4 = Mat::zeros(Size(4, 4), CV_8UC3);
	std::cout << "m4:\n" << m4 << std::endl;

	// create Matlab风格 - 5
	Mat m5 = Mat::ones(Size(4, 4), CV_8UC3);
	std::cout << "m5:\n" << m5 << std::endl;

	// clone and copyTo
	Mat m6 = m4.clone();
	std::cout << "m6:\n" << m6 << std::endl;
	Mat m7;
	m2.copyTo(m7);
	std::cout << "m7:\n" << m7 << std::endl;

	// c++ 11 支持的初始化
	Mat m8 = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	std::cout << "m8:\n" << m8 << std::endl;

}

void MatDemo::pixel_visit() {
	Mat src = imread("D:/images/lovely-girl.jpg");
	int w = src.cols;
	int h = src.rows;
	int cn = src.channels();

	//  重新定义灰度级别， 16个级别
	uchar table[256];
	for (int i = 0; i < 256; ++i)
		table[i] = (uchar)(16 * (i / 16));
	double t1 = getTickCount();
	// method 1
	//for (int row = 0; row < h; row++) {
	//	for (int col = 0; col < w*cn; col++) {
	//		int pv = src.at<uchar>(row, col);
	//		src.at<uchar>(row, col) = table[pv];
	//	}
	//}

	// method 2
	/*uchar* currentRow;
	for (int row = 0; row < h; row++) {
		currentRow = src.ptr<uchar>(row);
		for (int col = 0; col < w*cn; col++) {
			src.at<uchar>(row, col) = table[currentRow[col]];
		}
	}*/
	

	//// iterator
	//switch (cn)
	//{
	//	case 1:
	//	{
	//		MatIterator_<uchar> it, end;
	//		for (it = src.begin<uchar>(), end = src.end<uchar>(); it != end; ++it)
	//			*it = table[*it];
	//		break;
	//	}
	//	case 3:
	//	{
	//		MatIterator_<Vec3b> it, end;
	//		for (it = src.begin<Vec3b>(), end = src.end<Vec3b>(); it != end; ++it)
	//		{
	//			(*it)[0] = table[(*it)[0]];
	//			(*it)[1] = table[(*it)[1]];
	//			(*it)[2] = table[(*it)[2]];
	//		}
	//	}
	//}
	

	// use data
	uchar* image_data = src.data;
	for (int row = 0; row < h; row++) {
		for (int col = 0; col < w*cn; col++) {
			*image_data= table[*image_data];
			image_data++;
		}
	}
	double t2 = getTickCount();
	double t = ((t2 - t1) / getTickFrequency()) * 1000;
	ostringstream ss;
	ss << "Execute time : " << std::fixed << std::setprecision(2) << t << " ms ";
	std::cout << ss.str() << std::endl;
	imshow("result", src);
}

void MatDemo::suanshu_demo() {
	Mat src1 = imread(rootdir + "WindowsLogo.jpg");
	Mat src2 = imread(rootdir + "LinuxLogo.jpg");
	Mat add_result, sub_result, mul_result, div_result;
	add(src1, src2, add_result);
	subtract(src1, src2, sub_result);
	multiply(src1, src2, mul_result);
	divide(src1, src2, div_result);
	imshow("add_result", add_result);
	imshow("sub_result", sub_result);
	imshow("mul_result", mul_result);
	imshow("div_result", div_result);
}

void MatDemo::bitwise_demo() {
	Mat src1 = imread(rootdir + "WindowsLogo.jpg");
	Mat src2 = imread(rootdir + "LinuxLogo.jpg");
	Mat invert_result, and_result, or_result, xor_result;
	bitwise_not(src1, invert_result);
	bitwise_and(src1, src2, and_result);
	bitwise_or(src1, src2, or_result);
	bitwise_xor(src1, src2, xor_result);
	imshow("invert_result", invert_result);
	imshow("and_result", and_result);
	imshow("or_result", or_result);
	imshow("xor_result", xor_result);
}

void MatDemo::adjust_light(Mat &image) {
	Mat constant_img = Mat::zeros(image.size(), image.type());
	constant_img.setTo(Scalar(50, 50, 50));
	Mat darkMat, lightMat;
	// 亮度增强
	add(image, constant_img, lightMat);
	// 亮度降低
	subtract(image, constant_img, darkMat);
	// 显示
	imshow("lightMat", lightMat);
	imshow("darkMat", darkMat);
}

void MatDemo::adjust_contrast(Mat &image) {
	Mat constant_img = Mat::zeros(image.size(), CV_32FC3);
	constant_img.setTo(Scalar(0.8, 0.8, 0.8));
	Mat lowContrastMat, highContrastMat;
	// 低对比度
	multiply(image, constant_img, lowContrastMat, 1.0, CV_8U);
	// 亮度降低
	divide(image, constant_img, highContrastMat, 1.0, CV_8U);
	// 显示
	imshow("lowContrastMat", lowContrastMat);
	imshow("highContrastMat", highContrastMat);
}

void MatDemo::type_convert(Mat &image) {
	Mat f;
	image.convertTo(f, CV_32F);
	f = f / 255.0;
	imshow("f32", f);
}

void MatDemo::channels_demo(Mat &image) {
	std::vector<Mat> mv;
	split(image, mv);
	for (size_t t = 0; t < mv.size(); t++) {
		bitwise_not(mv[t], mv[t]);
	}
	Mat dst;
	merge(mv, dst);
	imshow("merge channels", dst);

	// 获取通道与混合
	int from_to[] = { 0,2, 1,1, 2,0 };
	mixChannels(&image, 1, &dst, 1, from_to, 3);
	imshow("mix channels", dst);
}

int main(int argc, char** argv) {
	//Mat src = imread(rootdir + "baboon.jpg");
	//imshow("input", src);
	//int cn = src.channels();
	//printf("image channels : %d \n", cn);
	//Mat gray;
	//cvtColor(src, gray, COLOR_BGR2GRAY);
	//cn = gray.channels();
	//printf("gray channels : %d \n", cn);
	
	// 通道分离与合并
	MatDemo md;
	md.pixel_visit();
	waitKey(0);
	return 0;
}