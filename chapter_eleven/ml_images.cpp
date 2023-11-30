#include <ml_images.h>
#include <iostream>

using namespace cv;
using namespace std;
string rootdir = "D:/opencv-4.2.0/opencv/sources/samples/data/";

void MLoperatorsDemo::kmeans_segmentation_demo(Mat &image) {
	Scalar colorTab[] = {
		Scalar(0, 0, 255),
		Scalar(0, 255, 0),
		Scalar(255, 0, 0),
		Scalar(0, 255, 255),
		Scalar(255, 0, 255)
	};

	int width = image.cols;
	int height = image.rows;
	int dims = image.channels();

	// 初始化定义
	int sampleCount = width*height;
	int clusterCount = 5;
	Mat labels;
	Mat centers;

	// RGB 数据转换到样本数据
	Mat sample_data = image.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// 运行K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	// 显示图像分割结果
	int index = 0;
	Mat result = Mat::zeros(image.size(), image.type());
	for (int row = 0; row < height; row++) {
		for (int col = 0; col < width; col++) {
			index = row*width + col;
			int label = labels.at<int>(index, 0);
			result.at<Vec3b>(row, col)[0] = colorTab[label][0];
			result.at<Vec3b>(row, col)[1] = colorTab[label][1];
			result.at<Vec3b>(row, col)[2] = colorTab[label][2];
		}
	}

	imshow("KMeans图像分割", result);
}

void MLoperatorsDemo::mainColorComponents(Mat &image) {
	int width = image.cols;
	int height = image.rows;
	int dims = image.channels();

	// 初始化定义
	int sampleCount = width*height;
	int clusterCount = 5;
	Mat labels;
	Mat centers;

	// RGB 数据转换到样本数据
	Mat sample_data = image.reshape(3, sampleCount);
	Mat data;
	sample_data.convertTo(data, CV_32F);

	// 运行K-Means
	TermCriteria criteria = TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 0.1);
	kmeans(data, clusterCount, labels, criteria, clusterCount, KMEANS_PP_CENTERS, centers);

	Mat card = Mat::zeros(Size(width, 50), CV_8UC3);
	vector<float> clusters(clusterCount);

	// 生成色卡比率
	for (int i = 0; i < labels.rows; i++) {
		clusters[labels.at<int>(i, 0)]++;
	}
	for (int i = 0; i < clusters.size(); i++) {
		clusters[i] = clusters[i] / sampleCount;
	}
	int x_offset = 0;

	// 绘制色卡
	for (int x = 0; x < clusterCount; x++) {
		Rect rect;
		rect.x = x_offset;
		rect.y = 0;
		rect.height = 50;
		rect.width = round(clusters[x] * width);
		x_offset += rect.width;
		int b = centers.at<float>(x, 0);
		int g = centers.at<float>(x, 1);
		int r = centers.at<float>(x, 2);
		rectangle(card, rect, Scalar(b, g, r), -1, 8, 0);
	}

	imshow("色卡生成", card);
}

void MLoperatorsDemo::knn_digit_train(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// 分割为5000个cells
	Mat images = Mat::zeros(5000, 400, CV_8UC1);
	Mat labels = Mat::zeros(5000, 1, CV_8UC1);

	int index = 0;
	Rect roi;
	roi.x = 0;
	roi.height = 1;
	roi.width = 400;
	for (int row = 0; row < 50; row++) {
		int label = row / 5;
		int offsety = row * 20;
		for (int col = 0; col < 100; col++) {
			int offsetx = col * 20;
			Mat digit = Mat::zeros(Size(20, 20), CV_8UC1);
			for (int sr = 0; sr < 20; sr++) {
				for (int sc = 0; sc < 20; sc++) {
					digit.at<uchar>(sr, sc) = gray.at<uchar>(sr + offsety, sc + offsetx);
				}
			}
			Mat one_row = digit.reshape(1, 1);
			roi.y = index;
			one_row.copyTo(images(roi));
			labels.at<uchar>(index, 0) = label;
			index++;
		}
	}
	printf("load sample hand-writing data...\n");

	// 转换为浮点数
	images.convertTo(images, CV_32FC1);
	labels.convertTo(labels, CV_32SC1);
	printf("load sample hand-writing data...\n");

	// 开始KNN训练
	printf("Start to knn train...\n");
	Ptr<ml::KNearest> knn = ml::KNearest::create();
	knn->setDefaultK(5);
	knn->setIsClassifier(true);
	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);
	knn->train(tdata);
	knn->save("D:/vcworkspaces/knn_knowledge.yml");
	printf("Finished KNN...\n");
}

void MLoperatorsDemo::knn_digit_test() {
	// real test it
	Mat t1 = imread(rootdir + "knn_01.png", IMREAD_GRAYSCALE);
	Mat t2 = imread(rootdir + "knn_02.png", IMREAD_GRAYSCALE);
	namedWindow("t1", WINDOW_FREERATIO);
	namedWindow("t2", WINDOW_FREERATIO);
	imshow("t1", t1);
	imshow("t2", t2);
	Mat m1, m2;
	resize(t1, m1, Size(20, 20));
	resize(t2, m2, Size(20, 20));
	Mat testdata = Mat::zeros(2, 400, CV_8UC1);
	Mat testlabels = Mat::zeros(2, 1, CV_32SC1);
	Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	Mat one = m1.reshape(1, 1);
	Mat two = m2.reshape(1, 1);
	one.copyTo(testdata(rect));
	rect.y = 1;
	two.copyTo(testdata(rect));
	testlabels.at<int>(0, 0) = 1;
	testlabels.at<int>(1, 0) = 2;
	testdata.convertTo(testdata, CV_32F);

	// 加载KNN分类器
	Ptr<ml::KNearest> knn = Algorithm::load<ml::KNearest>("D:/vcworkspaces/knn_knowledge.yml");
	Mat result;
	knn->findNearest(testdata, 5, result);
	for (int i = 0; i< result.rows; i++) {
		int predict = result.at<float>(i, 0);
		printf("knn t%d predict : %d, actual label ：%d \n", (i + 1), predict, testlabels.at<int>(i, 0));
	}
}

void MLoperatorsDemo::svm_digit_train(Mat &image) {
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);

	// 分割为5000个cells
	Mat images = Mat::zeros(5000, 400, CV_8UC1);
	Mat labels = Mat::zeros(5000, 1, CV_8UC1);

	int index = 0;
	Rect roi;
	roi.x = 0;
	roi.height = 1;
	roi.width = 400;
	for (int row = 0; row < 50; row++) {
		int label = row / 5;
		int offsety = row * 20;
		for (int col = 0; col < 100; col++) {
			int offsetx = col * 20;
			Mat digit = Mat::zeros(Size(20, 20), CV_8UC1);
			for (int sr = 0; sr < 20; sr++) {
				for (int sc = 0; sc < 20; sc++) {
					digit.at<uchar>(sr, sc) = gray.at<uchar>(sr + offsety, sc + offsetx);
				}
			}
			Mat one_row = digit.reshape(1, 1);
			roi.y = index;
			one_row.copyTo(images(roi));
			labels.at<uchar>(index, 0) = label;
			printf("index : %d, label : %d \n", index, label);
			index++;
		}
	}

	// 转换为浮点数
	images.convertTo(images, CV_32FC1);
	labels.convertTo(labels, CV_32SC1);
	printf("load sample hand-writing data...\n");

	Ptr<ml::SVM> svm = ml::SVM::create();
	svm->setGamma(0.02);
	svm->setC(0.5);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setType(ml::SVM::C_SVC);
	printf("Starting SVM Model Training...\n");
	Ptr<ml::TrainData> tdata = ml::TrainData::create(images, ml::ROW_SAMPLE, labels);
	svm->train(tdata);
	svm->save("D:/vcworkspaces/svm_knowledge.yml");
	printf("Finished Training SVM...\n");
}

void MLoperatorsDemo::svm_digit_test() {
	// real test it
	Mat t1 = imread(rootdir + "knn_01.png", IMREAD_GRAYSCALE);
	Mat t2 = imread(rootdir + "knn_02.png", IMREAD_GRAYSCALE);
	namedWindow("t1", WINDOW_FREERATIO);
	namedWindow("t2", WINDOW_FREERATIO);
	imshow("t1", t1);
	imshow("t2", t2);
	Mat m1, m2;
	resize(t1, m1, Size(20, 20));
	resize(t2, m2, Size(20, 20));
	Mat testdata = Mat::zeros(2, 400, CV_8UC1);
	Mat testlabels = Mat::zeros(2, 1, CV_32SC1);
	Rect rect;
	rect.x = 0;
	rect.y = 0;
	rect.height = 1;
	rect.width = 400;
	Mat one = m1.reshape(1, 1);
	Mat two = m2.reshape(1, 1);
	one.copyTo(testdata(rect));
	rect.y = 1;
	two.copyTo(testdata(rect));
	testlabels.at<int>(0, 0) = 1;
	testlabels.at<int>(1, 0) = 2;
	testdata.convertTo(testdata, CV_32F);

	// 加载SVM分类器
	Ptr<ml::SVM> svm = ml::StatModel::load<ml::SVM>("D:/vcworkspaces/svm_knowledge.yml");
	Mat result;
	svm->predict(testdata, result);
	for (int i = 0; i < result.rows; i++)
	{
		int predict = result.at<float>(i, 0);
		printf("svm t%d predict : %d, actual label ：%d \n", (i + 1), predict, testlabels.at<int>(i, 0));
	}
}

void MLoperatorsDemo::get_hog_descriptor(Mat &image, vector<float> &desc) {
	HOGDescriptor hog;
	int h = image.rows;
	int w = image.cols;
	float rate = 64.0 / w;
	Mat img, gray;
	resize(image, img, Size(64, int(rate*h)));
	cvtColor(img, gray, COLOR_BGR2GRAY);
	Mat result = Mat::zeros(Size(64, 128), CV_8UC1);
	result = Scalar(127);
	Rect roi;
	roi.x = 0;
	roi.width = 64;
	roi.y = (128 - gray.rows) / 2;
	roi.height = gray.rows;
	gray.copyTo(result(roi));
	hog.compute(result, desc, Size(8, 8), Size(0, 0));
}

void MLoperatorsDemo::train_ele_watch(std::string positive_dir, std::string negative_dir) {
	// 创建变量
	Mat trainData = Mat::zeros(Size(3780, 26), CV_32FC1);
	Mat labels = Mat::zeros(Size(1, 26), CV_32SC1);
	vector<string> images;
	glob(positive_dir, images);
	int pos_num = images.size();

	// 生成正负样本数据
	for (int i = 0; i < images.size(); i++) {
		Mat image = imread(images[i].c_str());
		vector<float> fv;
		get_hog_descriptor(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		for (int j = 0; j < fv.size(); j++) {
			trainData.at<float>(i, j) = fv[j];
		}
		labels.at<int>(i, 0) = 1;
	}

	images.clear();
	glob(negative_dir, images);
	for (int i = 0; i < images.size(); i++) {
		Mat image = imread(images[i].c_str());
		vector<float> fv;
		get_hog_descriptor(image, fv);
		printf("image path : %s, feature data length: %d \n", images[i].c_str(), fv.size());
		for (int j = 0; j < fv.size(); j++) {
			trainData.at<float>(i + pos_num, j) = fv[j];
		}
		labels.at<int>(i + pos_num, 0) = -1;
	}

	// 训练SVM仪表分类器
	printf("\n start SVM training... \n");
	Ptr< ml::SVM > svm = ml::SVM::create();
	svm->setKernel(ml::SVM::LINEAR);
	svm->setC(2.0);
	svm->setType(ml::SVM::C_SVC);
	svm->train(trainData, ml::ROW_SAMPLE, labels);
	clog << "...[done]" << endl;

	// save xml
	svm->save("D:/vcworkspaces/svm_hog_elec.yml");
}

void MLoperatorsDemo::hog_svm_detector_demo(Mat &image) {
	// 创建HOG与加载SVM训练数据
	HOGDescriptor hog;
	Ptr<ml::SVM> svm = ml::SVM::load("D:/vcworkspaces/svm_hog_elec.yml");
	Mat sv = svm->getSupportVectors();
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	// 构建detector
	vector<float> svmDetector;
	svmDetector.clear();
	svmDetector.resize(sv.cols + 1);
	for (int j = 0; j < sv.cols; j++) {
		svmDetector[j] = -sv.at<float>(0, j);
	}
	svmDetector[sv.cols] = (float)rho;
	hog.setSVMDetector(svmDetector);

	vector<Rect> objects;
	hog.detectMultiScale(image, objects, 0.1, Size(8, 8), Size(32, 32), 1.25);
	for (int i = 0; i < objects.size(); i++) {
		rectangle(image, objects[i], Scalar(0, 0, 255), 2, 8, 0);
	}
	namedWindow("SVM+HOG对象检测演示", WINDOW_FREERATIO);
	imshow("SVM+HOG对象检测演示", image);
}

int main(int argc, char** argv) {
	Mat image = imread(rootdir + "/elec_watch/test/scene_08.jpg");
	imshow("image", image);
	MLoperatorsDemo mler;
	// mler.train_ele_watch(rootdir + "/elec_watch/positive", rootdir + "/elec_watch/negative");
	mler.hog_svm_detector_demo(image);
	waitKey(0);
	destroyAllWindows();
	return 0;
}
