#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.5.4/opencv/sources/samples/data/";
void video_demo();
int main(int argc, char** argv) {
	Mat image = imread(rootdir + "lena.jpg", IMREAD_UNCHANGED);
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", image);
	waitKey(0);
	destroyAllWindows();
	video_demo();
	return 0;
}

void video_demo() {
	VideoCapture capture;
	capture.open(rootdir+"vtest.avi", CAP_FFMPEG);
	int height = capture.get(CAP_PROP_FRAME_HEIGHT);
	int width = capture.get(CAP_PROP_FRAME_WIDTH);
	double fps = capture.get(CAP_PROP_FPS);
	double count = capture.get(CAP_PROP_FRAME_COUNT);
	printf("height: %d, width: %d, fps: %.2f, count: %.2f \n", height, width, fps, count);

	VideoWriter writer("D:/output.avi", capture.get(CAP_PROP_FOURCC), fps, Size(width, height));
	Mat frame;
	while (true) {
		// ∂¡÷°
		bool ret = capture.read(frame);
		if (!ret) break;
		imshow("frame", frame);
		// ÃÌº”÷°¥¶¿Ì
		// .....
		writer.write(frame);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}
	capture.release();
	writer.release();
	waitKey(0);
	destroyAllWindows();
	return;
}