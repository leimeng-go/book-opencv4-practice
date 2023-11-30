#include <fastdnn.h>
#include <fstream>

string rootdir = "D:/opencv-4.5.4/opencv/sources/samples/data/";
string model_dir = "D:/projects/opencv_tutorial_data/models/";

std::vector<String> readClassNames(std::string label_map_file)
{
	std::vector<String> classNames;
	std::ifstream fp(label_map_file);
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

void OpenVINOInferenceModels::setup_test() {
	// 创建IE插件, 查询支持硬件设备
	ov::Core ie;
	vector<string> availableDevices = ie.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}
}

void OpenVINOInferenceModels::image_classification_demo(Mat &image) {
	String defect_labels[] = { "In","Sc","Cr","PS","RS","Pa" };
	std::string onnx_path = "D:/projects/opencv_tutorial_data/models/surface_defect_resnet18.onnx";

	// 加载模型
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnx_path, "GPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// 请求网络输入
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	ov::Shape tensor_shape = input_tensor.get_shape();
	size_t num_channels = tensor_shape[1];
	size_t h = tensor_shape[2];
	size_t w = tensor_shape[3];
	
	// 预处理
	cv::Mat blob_image;
	cv::resize(image, blob_image, cv::Size(w, h));
	cv::cvtColor(blob_image, blob_image, cv::COLOR_BGR2RGB);
	blob_image.convertTo(blob_image, CV_32F);
	blob_image = blob_image / 255.0;
	cv::subtract(blob_image, cv::Scalar(0.485, 0.456, 0.406), blob_image);
	cv::divide(blob_image, cv::Scalar(0.229, 0.224, 0.225), blob_image);

	// NCHW 设置输入图象数据
	size_t image_size = w * h;
	float* data = input_tensor.data<float>();
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * w + col] = blob_image.at<cv::Vec3f>(row, col)[ch];
			}
		}
	}

	// 推理
	infer_request.infer();

	//返回结果
	auto output = infer_request.get_output_tensor();
	const float* prob = (float*)output.data();
	float max = prob[0];
	int max_index = 0;

	for (int i = 1; i < 6; i++) {
		if (max < prob[i]) {
			max = prob[i];
			max_index = i;
		}
	}

	std::cout << "class index : " << max_index << std::endl;
	std::cout << "class name : " << defect_labels[max_index] << std::endl;
	cv::putText(image, defect_labels[max_index], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);

	imshow("图像分类演示", image);
}

void OpenVINOInferenceModels::yolov5_demo() {
	std::string onnx_path = "D:/python/yolov5-7.0/yolov5s.onnx";
	std::string labels_file = "D:/python/yolov5-7.0/classes.txt";
	std::vector<std::string> class_names = readClassNames(labels_file);

	// 加载模型
	ov::Core ie;
	ov::CompiledModel compiled_model = ie.compile_model(onnx_path, "CPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// 预处理输入数据 - 格式化操作
	VideoCapture cap;
	cap.open("D:/bird_test/Pexels_Videos_2670.mp4");
	if (!cap.isOpened()) {
		cout << "Exit! fails to open!" << endl;
		return;
	}

	// 获取输入节点tensor, NCHW-H, W
	auto input_image_tensor = infer_request.get_input_tensor();
	int input_h = input_image_tensor.get_shape()[2];
	int input_w = input_image_tensor.get_shape()[3]; 

	// 获取输出格式, boxes number, 5 + 类别数目
	auto output_tensor = infer_request.get_output_tensor();
	int out_rows = output_tensor.get_shape()[1]; 
	int out_cols = output_tensor.get_shape()[2];
	cv::namedWindow("YOLOv5-7.x + OpenVINO2023 演示", cv::WINDOW_NORMAL);

	//连续采集处理循环
	Mat frame;
	while (true) {
		bool ret = cap.read(frame);
		if (!ret) {
			break;
		}

		int64 start = cv::getTickCount();
		int w = frame.cols;
		int h = frame.rows;
		int _max = std::max(h, w);
		cv::Mat image = cv::Mat::zeros(cv::Size(_max, _max), CV_8UC3);
		cv::Rect roi(0, 0, w, h);
		frame.copyTo(image(roi));

		float x_factor = image.cols / input_w;
		float y_factor = image.rows / input_h;

		cv::Mat blob_image;
		resize(image, blob_image, cv::Size(input_w, input_h));
		blob_image.convertTo(blob_image, CV_32F);
		blob_image = blob_image / 255.0;

		// NCHW 图象数据填充到tensor对象
		size_t image_size = input_w * input_h;
		float* data = input_image_tensor.data<float>();
		for (size_t row = 0; row < input_h; row++) {
			for (size_t col = 0; col < input_w; col++) {
				for (size_t ch = 0; ch < 3; ch++) {
					data[image_size*ch + row * input_w + col] = blob_image.at<cv::Vec3f>(row, col)[ch];
				}
			}
		}

		// 执行推理计算
		infer_request.infer();

		// 获得推理结果
		const ov::Tensor& output_tensor = infer_request.get_output_tensor();

		// 解析推理结果，YOLOv5 output format: cx,cy,w,h,score + 类别数目
		cv::Mat det_output(out_rows, out_cols, CV_32F, (float*)output_tensor.data());

		std::vector<cv::Rect> boxes;
		std::vector<int> classIds;
		std::vector<float> confidences;

		for (int i = 0; i < det_output.rows; i++) {
			float confidence = det_output.at<float>(i, 4);
			if (confidence < 0.25) {
				continue;
			}
			cv::Mat classes_scores = det_output.row(i).colRange(5, 85);
			cv::Point classIdPoint;
			double score;
			minMaxLoc(classes_scores, 0, &score, 0, &classIdPoint);

			// 置信度 0～1之间
			if (score > 0.25)
			{
				float cx = det_output.at<float>(i, 0);
				float cy = det_output.at<float>(i, 1);
				float ow = det_output.at<float>(i, 2);
				float oh = det_output.at<float>(i, 3);
				int x = static_cast<int>((cx - 0.5 * ow) * x_factor);
				int y = static_cast<int>((cy - 0.5 * oh) * y_factor);
				int width = static_cast<int>(ow * x_factor);
				int height = static_cast<int>(oh * y_factor);
				cv::Rect box;
				box.x = x;
				box.y = y;
				box.width = width;
				box.height = height;

				boxes.push_back(box);
				classIds.push_back(classIdPoint.x);
				confidences.push_back(score);
			}
		}
		// NMS
		std::vector<int> indexes;
		cv::dnn::NMSBoxes(boxes, confidences, 0.25, 0.5, indexes);
		for (size_t i = 0; i < indexes.size(); i++) {
			int index = indexes[i];
			int idx = classIds[index];
			cv::rectangle(frame, boxes[index], cv::Scalar(0, 0, 255), 2, 8);
			cv::rectangle(frame, cv::Point(boxes[index].tl().x, boxes[index].tl().y - 20),
				cv::Point(boxes[index].br().x, boxes[index].tl().y), cv::Scalar(0, 255, 255), -1);
			cv::putText(frame, class_names[idx], cv::Point(boxes[index].tl().x, boxes[index].tl().y - 10), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0, 0, 0));
		}

		// 计算FPS render it
		float t = (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
		// cout << "Infer time(ms): " << t * 1000 << "ms; Detections: " << indexes.size() << endl;
		putText(frame, cv::format("FPS: %.2f", 1.0 / t), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255, 0, 0), 2, 8);
		cv::imshow("YOLOv5-7.x + OpenVINO2023 演示", frame);
		char c = cv::waitKey(1);
		if (c == 27) { // ESC
			break;
		}
	}
}

void OpenVINOInferenceModels::unet_demo(Mat &image) {
	// 创建IE插件, 查询支持硬件设备
	ov::Core ie;
	vector<string> availableDevices = ie.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
	}

	//  加载检测模型
	std::string onnx_path = "D:/python/pytorch_tutorial/defect_unet/unet_road.onnx";
	ov::CompiledModel compiled_model = ie.compile_model(onnx_path, "CPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();

	// 请求网络输入
	ov::Tensor input_tensor = infer_request.get_input_tensor();
	ov::Shape tensor_shape = input_tensor.get_shape();
	size_t num_channels = tensor_shape[1];
	size_t h = tensor_shape[2];
	size_t w = tensor_shape[3];

	Mat gray, blob_image;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	resize(gray, blob_image, Size(w, h));
	blob_image.convertTo(blob_image, CV_32F);
	blob_image = blob_image / 255.0;

	// NCHW
	float* image_data = input_tensor.data<float>();
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			image_data[row * w + col] = blob_image.at<float>(row, col);
		}
	}

	// 执行预测
	infer_request.infer();

	// 获取输出数据
	auto output_tensor = infer_request.get_output_tensor();
	const float* detection = (float*)output_tensor.data();
	ov::Shape out_shape = output_tensor.get_shape();
	const int out_c = out_shape[1];
	const int out_h = out_shape[2];
	const int out_w = out_shape[3];
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
	imshow("input", image);
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(result, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point());
	cv::drawContours(image, contours, -1, cv::Scalar(0, 0, 255), -1, 8);
	imshow("OpenVINO+UNet", image);
}

void OpenVINOInferenceModels::landmark_demo(std::string landmark_model_path) {
	//  加载检测模型
	ov::Core ie;
	auto model = ie.read_model(landmark_model_path);
	// 设置动态输入
	model->reshape({ 1, 3, -1, -1 });
	// 编译关联设备
	ov::CompiledModel compiled_model = ie.compile_model(model, "CPU");
	ov::InferRequest infer_request = compiled_model.create_infer_request();
	VideoCapture cap;
	cap.open("D:/images/video/test_pfld.mp4");
	//连续采集处理循环
	Mat frame, blob_image;
	while (true) {
		bool ret = cap.read(frame);
		if (!ret) {
			break;
		}

		int64 start = cv::getTickCount();
		size_t input_w = frame.cols;
		size_t input_h = frame.rows;
		// NCHW 图象数据填充到tensor对象
		ov::Tensor input_data = ov::Tensor(ov::element::f32, { 1, 3, 544, 960 });
		blob_image = frame.clone();
		blob_image.convertTo(blob_image, CV_32F);
		blob_image = blob_image / 255.0;

		size_t image_size = input_w * input_h;
		float* data = input_data.data<float>();
		for (size_t row = 0; row < input_h; row++) {
			for (size_t col = 0; col < input_w; col++) {
				for (size_t ch = 0; ch < 3; ch++) {
					data[image_size*ch + row * input_w + col] = blob_image.at<cv::Vec3f>(row, col)[ch];
				}
			}
		}

		// 执行推理计算
		infer_request.set_input_tensor(input_data);
		infer_request.infer();

		// 获得推理结果
		const ov::Tensor& loc_tensor = infer_request.get_tensor("loc");
		const ov::Tensor& conf_tensor = infer_request.get_tensor("conf");
		ov::Shape loc_shape = loc_tensor.get_shape();
		ov::Shape conf_shape = conf_tensor.get_shape();
		const int out_rows = loc_shape[0];
		const int out_cols = loc_shape[1];
		std::cout << "loc_tensor:" << out_rows << " x " << out_cols << std::endl;
		std::cout << "conf_tensor:" << conf_shape[0] << " x " << conf_shape[1] << std::endl;
		const float* loc_data = (float*)loc_tensor.data();
		const float* conf_data = (float*)conf_tensor.data();
		for (int row = 0; row < out_rows; row++) {
			float score = conf_data[row * 2+1];
			std::cout << "score:" << score << std::endl;
			if (score > 0.60) {
				for (int col = 0; col < out_cols; col++) {
					std::cout << "loc_data:" << loc_data[row*out_cols+col] << std::endl;
				}
			}
		}
		break;
	}
}

int main(int argc, char** argv) {
	// Mat image = imread(rootdir + "321.jpg");
	// Mat image = imread("D:/images/test4.png");
	OpenVINOInferenceModels  openvino_models;
	openvino_models.yolov5_demo();
	waitKey(0);
	destroyAllWindows();
	return 0;
}