#include <dnn_demo.h>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.2.0/opencv/sources/samples/data/";
// please download models from 
// https://github.com/gloomyfish1998/opencv_tutorial
string model_dir = "D:/projects/opencv_tutorial/data/models/";

String objNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

std::vector<String> readClassNames()
{
	std::vector<String> classNames;

	std::ifstream fp(model_dir + "googlenet/classification_classes_ILSVRC2012.txt");
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}
void DeepNeuralNetOps::image_classification(Mat &image) {
	std::string weight_path = model_dir + "googlenet/bvlc_googlenet.caffemodel";
	std::string config_path = model_dir + "googlenet/bvlc_googlenet.prototxt";
	Net net = readNetFromCaffe(config_path, weight_path);
	if (net.empty()) {
		printf("read caffe model data failure...\n");
		return;
	}
	vector<String> labels = readClassNames();
	Mat inputBlob = blobFromImage(image, 1.0, Size(224, 224), Scalar(104, 117, 123), false, false);

	// 执行图像分类
	Mat prob;
	net.setInput(inputBlob);
	prob = net.forward();
	vector<double> times;
	double time = net.getPerfProfile(times);
	float ms = (time * 1000) / getTickFrequency();
	printf("current inference time : %.2f ms \n", ms);

	// 得到最可能分类输出
	Mat probMat = prob.reshape(1, 1);
	Point classNumber;
	double classProb;
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	int classidx = classNumber.x;
	printf("\n current image classification : %s, possible : %.2f", labels.at(classidx).c_str(), classProb);

	// 显示文本
	putText(image, labels.at(classidx), Point(20, 40), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8);
	imshow("图像分类演示", image);
}

void DeepNeuralNetOps::ssd_demo(Mat &image) {
	std::string ssd_config = model_dir + "ssd/MobileNetSSD_deploy.prototxt";
	std::string ssd_weight = model_dir + "ssd/MobileNetSSD_deploy.caffemodel";
	Net net = readNetFromCaffe(ssd_config, ssd_weight);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);
	Mat blobImage = blobFromImage(image, 0.007843,
		Size(300, 300),
		Scalar(127.5, 127.5, 127.5), true, false);
	printf("blobImage height : %d, width: %d\n", blobImage.size[2], blobImage.size[3]);

	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * image.cols;
			float tl_y = detectionMat.at<float>(i, 4) * image.rows;
			float br_x = detectionMat.at<float>(i, 5) * image.cols;
			float br_y = detectionMat.at<float>(i, 6) * image.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(image, object_box, Scalar(0, 0, 255), 2, 8, 0);
			putText(image, format(" confidence %.2f, %s", confidence, objNames[objIndex].c_str()),
				Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2, 8);
		}
	}
	imshow("SSD对象检测", image);

}

std::map<int, string> readcocoLabels()
{
	std::map<int, string> labelNames;
	std::ifstream fp(model_dir + "faster_rcnn/mscoco_label_map.pbtxt");
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	string one_line;
	string display_name;
	while (!fp.eof())
	{
		std::getline(fp, one_line);
		std::size_t found = one_line.find("id:");
		if (found != std::string::npos) {
			int index = found;
			string id = one_line.substr(index + 4, one_line.length() - index);

			std::getline(fp, display_name);
			std::size_t  found = display_name.find("display_name:");

			index = found + 15;
			string name = display_name.substr(index, display_name.length() - index);
			name = name.replace(name.length() - 1, name.length(), "");
			// printf("id : %d, name: %s \n", stoi(id.c_str()), name.c_str());
			labelNames[stoi(id)] = name;
		}
	}
	fp.close();
	return labelNames;
}

void DeepNeuralNetOps::faster_rcnn_demo(Mat &image) {
	// 加载网络
	std::string faster_rcnn_config = model_dir + "faster_rcnn/faster-rcnn.pbtxt";
	std::string faster_rcnn_weight = model_dir + "faster_rcnn/frozen_inference_graph.pb";
	Net net = readNetFromTensorflow(faster_rcnn_weight, faster_rcnn_config);
	map<int, string> names = readcocoLabels();

	// 设置输入Blob
	Mat blobImage = blobFromImage(image, 1.0,
		Size(800, 600),
		Scalar(0, 0, 0), true, false);
	printf("blobImage height : %d, width: %d\n", blobImage.size[2], blobImage.size[3]);
	net.setInput(blobImage);

	// 推理
	Mat detection = net.forward();
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);

	// 解析输出
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.5;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float tl_x = detectionMat.at<float>(i, 3) * image.cols;
			float tl_y = detectionMat.at<float>(i, 4) * image.rows;
			float br_x = detectionMat.at<float>(i, 5) * image.cols;
			float br_y = detectionMat.at<float>(i, 6) * image.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(image, object_box, Scalar(0, 0, 255), 2, 8, 0);
			map<int, string>::iterator it = names.find(objIndex+1);
			printf("id : %d, display name : %s \n", objIndex + 1, (it->second).c_str());
			putText(image, format(" confidence %.2f, %s", confidence, (it->second).c_str()),
				Point(tl_x - 10, tl_y - 5), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 1, 8);
		}
	}
	imshow("Faster-RCNN对象检测", image);
}

void DeepNeuralNetOps::yolo_demo(Mat &image) {
	string yolov4_model = model_dir + "yolov4-leaky-416.weights";
	string yolov4_config = model_dir + "yolov4-leaky-416.cfg";

	vector<string> classNamesVec;
	ifstream classNamesFile(model_dir + "object_detection_classes_yolov4.txt");
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			classNamesVec.push_back(className);
	}
	// 加载YOLOv4
	Net net = readNetFromDarknet(yolov4_config, yolov4_model);
	std::vector<String> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}

	// 设置输入
	Mat inputBlob = blobFromImage(image, 1 / 255.F, Size(416, 416), Scalar(), true, false);
	net.setInput(inputBlob);

	// 预测
	std::vector<Mat> outs;
	net.forward(outs, outNames);

	vector<Rect> boxes;
	vector<int> classIds;
	vector<float> confidences;
	for (size_t i = 0; i<outs.size(); ++i)
	{
		// 解析与合并各输出层的预测结果
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5)
			{
				int centerX = (int)(data[0] * image.cols);
				int centerY = (int)(data[1] * image.rows);
				int width = (int)(data[2] * image.cols);
				int height = (int)(data[3] * image.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// 非最大抑制与输出
	vector<int> indices;
	NMSBoxes(boxes, confidences, 0.5, 0.2, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		String className = classNamesVec[classIds[idx]];
		putText(image, className.c_str(), box.tl(), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2, 8);
		rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
	}
	imshow("YOLOv4-Detections", image);
}

void postENetProcess(Mat &score, Mat &mask) {
	const int rows = score.size[2];
	const int cols = score.size[3];
	const int chns = score.size[1];
	Mat maxVal = Mat::zeros(rows, cols, CV_32FC1);
	for (int ch = 1; ch < chns; ch++)
	{
		for (int row = 0; row < rows; row++)
		{
			const float *ptrScore = score.ptr<float>(0, ch, row);
			uchar *ptrMaxCl = mask.ptr<uchar>(row);
			float *ptrMaxVal = maxVal.ptr<float>(row);
			for (int col = 0; col < cols; col++)
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col];
					ptrMaxCl[col] = (uchar)ch;
				}
			}
		}
	}
	normalize(mask, mask, 0, 255, NORM_MINMAX);
	applyColorMap(mask, mask, COLORMAP_HSV);
}

void DeepNeuralNetOps::enet_demo(Mat &image) {
	// 加载网络
	Net net = readNetFromTorch(model_dir + "enet/model-best.net");
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// 设置输入
	Mat blob = blobFromImage(image, 0.00392, Size(512, 256), Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	// 推理预测
	Mat score = net.forward();
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);

	// 解析输出与显示
	Mat mask = Mat::zeros(256, 512, CV_8UC1);
	postENetProcess(score, mask);
	resize(mask, mask, image.size());
	Mat dst;
	addWeighted(image, 0.8, mask, 0.2, 0, dst);
	imshow("ENet道路分割演示", dst);
}

void DeepNeuralNetOps::style_transfer_demo(Mat &image) {
	Net net = readNetFromTorch(model_dir + "fast_style/candy.t7");
	Mat blobImage = blobFromImage(image, 1.0,
		image.size(),
		Scalar(103.939, 116.779, 123.68), false, false);

	net.setInput(blobImage);
	Mat out = net.forward();
	vector<double> layersTimings;
	double freq = getTickFrequency() / 1000;
	double time = net.getPerfProfile(layersTimings) / freq;
	printf("execute time : %.2f ms\n", time);
	int ch = out.size[1];
	int h = out.size[2];
	int w = out.size[3];
	Mat result = Mat::zeros(Size(w, h), CV_32FC3);
	float* data = out.ptr<float>();

	// decode 4-d Mat object
	for (int c = 0; c < ch; c++) {
		for (int row = 0; row < h; row++) {
			for (int col = 0; col < w; col++) {
				result.at<Vec3f>(row, col)[c] = *data++;
			}
		}
	}

	// 整合结果输出
	printf("channels : %d, height: %d, width: %d \n", ch, h, w);
	add(result, Scalar(103.939, 116.779, 123.68), result);
	normalize(result, result, 0, 1.0, NORM_MINMAX);

	// 中值滤波
	medianBlur(result, result, 5);
	imshow("风格迁移演示", result);
}

void decode(const Mat& scores, const Mat& geometry, float scoreThresh,
	std::vector<RotatedRect>& detections, std::vector<float>& confidences)
{
	detections.clear();
	CV_Assert(scores.dims == 4); CV_Assert(geometry.dims == 4); CV_Assert(scores.size[0] == 1);
	CV_Assert(geometry.size[0] == 1); CV_Assert(scores.size[1] == 1); CV_Assert(geometry.size[1] == 5);
	CV_Assert(scores.size[2] == geometry.size[2]); CV_Assert(scores.size[3] == geometry.size[3]);

	const int height = scores.size[2];
	const int width = scores.size[3];
	for (int y = 0; y < height; ++y)
	{
		const float* scoresData = scores.ptr<float>(0, 0, y);
		const float* x0_data = geometry.ptr<float>(0, 0, y);
		const float* x1_data = geometry.ptr<float>(0, 1, y);
		const float* x2_data = geometry.ptr<float>(0, 2, y);
		const float* x3_data = geometry.ptr<float>(0, 3, y);
		const float* anglesData = geometry.ptr<float>(0, 4, y);
		for (int x = 0; x < width; ++x)
		{
			float score = scoresData[x];
			if (score < scoreThresh)
				continue;

			// Decode a prediction.
			// Multiple by 4 because feature maps are 4 time less than input image.
			float offsetX = x * 4.0f, offsetY = y * 4.0f;
			float angle = anglesData[x];
			float cosA = std::cos(angle);
			float sinA = std::sin(angle);
			float h = x0_data[x] + x2_data[x];
			float w = x1_data[x] + x3_data[x];

			Point2f offset(offsetX + cosA * x1_data[x] + sinA * x2_data[x],
				offsetY - sinA * x1_data[x] + cosA * x2_data[x]);
			Point2f p1 = Point2f(-sinA * h, -cosA * h) + offset;
			Point2f p3 = Point2f(-cosA * w, sinA * w) + offset;
			RotatedRect r(0.5f * (p1 + p3), Size2f(w, h), -angle * 180.0f / (float)CV_PI);
			detections.push_back(r);
			confidences.push_back(score);
		}
	}
}

void DeepNeuralNetOps::text_detection_demo(Mat &image) {
	float confThreshold = 0.5;
	float nmsThreshold = 0.4;
	int inpWidth = 320;
	int inpHeight = 320;

	// Load network.
	Net net = readNet(model_dir + "east/frozen_east_text_detection.pb");

	std::vector<Mat> outs;
	std::vector<std::string> outNames = net.getUnconnectedOutLayersNames();
	for (int i = 0; i < outNames.size(); i++) {
		printf("output layer name : %s\n", outNames[i].c_str());
	}

	Mat blob;
	blobFromImage(image, blob, 1.0, Size(inpWidth, inpHeight), Scalar(123.68, 116.78, 103.94), true, false);
	net.setInput(blob);
	net.forward(outs, outNames);

	Mat geometry = outs[0]; // RBOX
	Mat scores = outs[1]; // Scores

	// 解析输出
	std::vector<RotatedRect> boxes;
	std::vector<float> confidences;
	decode(scores, geometry, confThreshold, boxes, confidences);

	// 非最大抑制
	std::vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

	// 绘制检测框
	Point2f ratio((float)image.cols / inpWidth, (float)image.rows / inpHeight);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		RotatedRect& box = boxes[indices[i]];

		Point2f vertices[4];
		box.points(vertices);
		for (int j = 0; j < 4; ++j)
		{
			vertices[j].x *= ratio.x;
			vertices[j].y *= ratio.y;
		}
		for (int j = 0; j < 4; ++j)
			line(image, vertices[j], vertices[(j + 1) % 4], Scalar(255, 0, 0), 2);
	}

	// 显示信息
	std::vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	std::string label = format("Inference time: %.2f ms", t);
	putText(image, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0));
	imshow("场景文字检测", image);
}

void face_detect(Mat &image, Net &net) {
	int h = image.rows;
	int w = image.cols;
	cv::Mat inputBlob = cv::dnn::blobFromImage(image, 1.0, cv::Size(300, 300),
		Scalar(104.0, 177.0, 123.0), false, false);

	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confidence = detectionMat.at<float>(i, 2);

		if (confidence > 0.125)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * w);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * h);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * w);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * h);

			cv::rectangle(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0),
				2, 8);
		}
	}
	imshow("人脸检测演示", image);
}

void DeepNeuralNetOps::face_detection_demo(Mat &image, bool tf) {
	const std::string caffe_config = model_dir + "face_detector/deploy.prototxt";
	const std::string caffe_weight = model_dir + "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";

	const std::string tf_config = model_dir + "face_detector/opencv_face_detector.pbtxt";
	const std::string tf_weight = model_dir + "face_detector/opencv_face_detector_uint8.pb";

	Net net;
	if (tf) {
		net = cv::dnn::readNetFromTensorflow(tf_weight, tf_config);
	}
	else {
		net = cv::dnn::readNetFromCaffe(caffe_config, caffe_weight);
	}
	face_detect(image, net);
}

void DeepNeuralNetOps::cam_face_detection_demo(bool tf) {
	const std::string caffe_config = model_dir + "face_detector/deploy.prototxt";
	const std::string caffe_weight = model_dir + "face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel";

	const std::string tf_config = model_dir + "face_detector/opencv_face_detector.pbtxt";
	const std::string tf_weight = model_dir + "face_detector/opencv_face_detector_uint8.pb";

	Net net;
	if (tf) {
		net = cv::dnn::readNetFromTensorflow(tf_weight, tf_config);
	}
	else {
		net = cv::dnn::readNetFromCaffe(caffe_config, caffe_weight);
	}
	VideoCapture capture(0);
	Mat frame;
	while (true) {
		bool ret = capture.read(frame);
		if (frame.empty()) {
			break;
		}
		face_detect(frame, net);
		char c = waitKey(1);
		if (c == 27) {
			break;
		}
	}
}

int main(int argc, char** argv) {
	Mat image = imread("D:/images/yige.png");
	imshow("image", image);
	DeepNeuralNetOps ops;
	ops.style_transfer_demo(image);
	imwrite("D:/result.png", image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
