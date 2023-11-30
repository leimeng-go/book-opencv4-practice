#include <cuda_demo.h>
#include <fstream>

// please download models from 
// https://github.com/gloomyfish1998/opencv_tutorial
string rootdir = "D:/opencv-4.8.0/opencv/book_images/";
string model_dir = "D:/projects/opencv_tutorial_data/models/";

void CUDASpeedUpDemo::queryDeviceInfo() {
	cuda::printCudaDeviceInfo(cuda::getDevice());
	int count = getCudaEnabledDeviceCount();
	printf("GPU Device count %d \n", count);
}

void CUDASpeedUpDemo::videoAnalysis() {
	VideoCapture cap;
	cap.open("D:/images/video/vtest.avi");
	auto mog = cuda::createBackgroundSubtractorMOG2();
	Mat frame;
	GpuMat d_frame, d_fgmask, d_bgimg;
	Mat fg_mask, bgimg, fgimg;
	namedWindow("input", WINDOW_AUTOSIZE);
	namedWindow("background", WINDOW_AUTOSIZE);
	namedWindow("mask", WINDOW_AUTOSIZE);
	Mat se = cv::getStructuringElement(MORPH_RECT, Size(5, 5));
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(frame);
		if (!ret) break;

		// 背景分析
		d_frame.upload(frame);
		mog->apply(d_frame, d_fgmask);
		mog->getBackgroundImage(d_bgimg);

		// 形态学操作
		auto morph_filter = cuda::createMorphologyFilter(MORPH_OPEN, d_fgmask.type(), se);
		morph_filter->apply(d_fgmask, d_fgmask);

		// download from GPU Mat
		d_bgimg.download(bgimg);
		d_fgmask.download(fg_mask);

		// 计算FPS
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(frame, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);

		imshow("input", frame);
		imshow("background", bgimg);
		imshow("mask", fg_mask);

		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

}

void CUDASpeedUpDemo::epf() {
	VideoCapture cap;
	cap.open("D:/images/video/example_dsh.mp4");
	Mat frame, result;
	GpuMat image;
	GpuMat dst;
	while (true) {
		int64 start = getTickCount();
		bool ret = cap.read(frame);
		if (!ret) break;
		image.upload(frame);
		cuda::cvtColor(image, image, COLOR_BGR2BGRA);
		cuda::bilateralFilter(image, dst, 0, 50, 5);
		dst.download(result);
		// cv::bilateralFilter(frame, result, 0, 100, 10);
		double fps = getTickFrequency() / (getTickCount() - start);
		putText(result, format("FPS: %.2f", fps), Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
		imshow("CUDA版本双边滤波", result);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}

}

void CUDASpeedUpDemo::faceDetection() {
	String modelBinary = model_dir + "face_detector/opencv_face_detector_uint8.pb";
	String modelDesc = model_dir + "face_detector/opencv_face_detector.pbtxt";
	dnn::Net net = readNetFromTensorflow(modelBinary, modelDesc);
	net.setPreferableTarget(DNN_TARGET_CUDA);
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	//net.setPreferableTarget(DNN_TARGET_CPU); 
	//net.setPreferableBackend(DNN_BACKEND_OPENCV);

	if (net.empty())
	{
		printf("could not load net...\n");
		return;
	}

	// 打开摄像头
	// VideoCapture capture(0);
	VideoCapture capture("D:/images/video/example_dsh.mp4");
	if (!capture.isOpened()) {
		printf("could not load camera...\n");
		return;
	}

	Mat frame;
	while (capture.read(frame)) {
		int64 start = getTickCount();
		if (frame.empty())
		{
			break;
		}

		// 输入数据调整
		Mat inputBlob = blobFromImage(frame, 1.0,
			Size(300, 300), Scalar(104.0, 177.0, 123.0), false, false);
		net.setInput(inputBlob, "data");

		// 人脸检测
		Mat detection = net.forward("detection_out");
		vector<double> layersTimings;
		double freq = getTickFrequency() / 1000;
		double time = net.getPerfProfile(layersTimings) / freq;
		Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

		ostringstream ss;
		for (int i = 0; i < detectionMat.rows; i++)
		{
			// 置信度 0～1之间
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > 0.5)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);
				Rect box(x1, y1, x2 - x1, y2 - y1);
				rectangle(frame, box, Scalar(0, 255, 0));
			}
		}
		float fps = getTickFrequency() / (getTickCount() - start);
		ss.str("");
		ss << "FPS: " << fps << " ; inference time: " << time << " ms";
		putText(frame, ss.str(), Point(300, 20), 0, 0.75, Scalar(0, 0, 255), 2, 8);
		imshow("人脸检测", frame);
		if (waitKey(1) >= 0) break;
	}
}

int main(int argc, char** argv) {
	/*
	Mat image = imread(rootdir + "lena.jpg");
	imshow("image", image);
	Mat gray;
	cuda::GpuMat gmat, gpu_gray;
	gmat.upload(image);
	cuda::cvtColor(gmat, gpu_gray, COLOR_BGR2GRAY);
	gpu_gray.download(gray);
	imshow("gray", gray);
	*/
	CUDASpeedUpDemo cuda_hepler;
	cuda_hepler.epf();
	waitKey(0);
	destroyAllWindows();
	return 0;
}