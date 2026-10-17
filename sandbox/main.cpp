#include <opencv2/core/ocl.hpp>
#include "basicFunctions.h"
#include "stabilizationFunctions.h"
//#include "wienerFilter.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define NCoef 10
#define DCgain 4

#define Ntap 31


void checkUmatFrames_(UMat uOldGray, UMat uGray)
{
 	// Убедимся в одинаковых типах
	if (uOldGray.type() != uGray.type()) {
		cerr << "func-Type mismatch! uOldGray: " << uOldGray.type() 
			<< ", uGray: " << uGray.type() << endl;
		uGray.convertTo(uGray, uOldGray.type());
	}
	
	// Убедимся в одинаковых размерах
	if (uOldGray.size() != uGray.size()) {
		cerr << "func-Size mismatch!" << endl;
		cv::resize(uGray, uGray, uOldGray.size());
	}
	
	// Отладочный вывод
	cout << "func-Types - Old: " << uOldGray.type() 
		<< ", New: " << uGray.type() 
		<< " (CV_8UC1 = " << CV_8UC1 << ")" << endl;
}



void checkOpenCLStatus() {
    // Проверка доступности OpenCL
    if (cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL is available: YES" << std::endl;
        
        // Проверить, используется ли OpenCL
        if (cv::ocl::useOpenCL()) {
            std::cout << "OpenCL is enabled: YES" << std::endl;
            
            // Получить контекст OpenCL
            cv::ocl::Context ctx = cv::ocl::Context::getDefault();
            if (!ctx.empty()) {
                std::cout << "OpenCL context created: YES" << std::endl;
                
                // Получить устройство
                cv::ocl::Device device = cv::ocl::Device::getDefault();
                std::cout << "OpenCL Device: " << device.name() << std::endl;
                std::cout << "Device Type: ";
                if (device.type() == cv::ocl::Device::TYPE_CPU)
                    std::cout << "CPU" << std::endl;
                else if (device.type() == cv::ocl::Device::TYPE_GPU)
                    std::cout << "GPU" << std::endl;
                else if (device.type() == cv::ocl::Device::TYPE_ACCELERATOR)
                    std::cout << "Accelerator" << std::endl;
                else
                    std::cout << "Unknown" << std::endl;
                    
                std::cout << "Device Vendor: " << device.vendorName() << std::endl;
                std::cout << "Device Version: " << device.driverVersion() << std::endl;
                std::cout << "Compute Units: " << device.maxComputeUnits() << std::endl;
            }
        } else {
            std::cout << "OpenCL is enabled: NO" << std::endl;
        }
    } else {
        std::cout << "OpenCL is available: NO" << std::endl;
    }
}


int main()
{
	checkOpenCLStatus();
	int outputResolution = 1000;
	TransformParam noiseIn = { 0.0, 0.0, 0.0 };
	vector <TransformParam> noiseOut(2);
	for (int i = 0; i < noiseOut.size();i++)
		noiseOut[i] = {0.0, 0.0, 0.0};
	
	cv::Mat TShake(2, 3, CV_64F);

	vector <TransformParam> X(1+NCoef), Y(1 + NCoef);

	// Автоматическое создание папок
	vector <std::string> folderPath(4); 
	folderPath[0] = "./OutputVideos";
	folderPath[1] = "./OutputResults";
	folderPath[2] = "./SourceVideos";
	folderPath[3] = "./SourceVideosAuto";
    for (int tmp = 0; tmp < folderPath.size(); tmp++)
	{
		if (!fs::exists(folderPath[tmp])) {
			if (!fs::create_directory(folderPath[tmp])) {
				std::cerr << "Failed to create directory!" << std::endl;
				return -1;
			}
		}
	}
	
	// Создадим массив случайных цветов для цветов характерных точек
	vector<cv::Scalar> colors(1000);
	RNG rng;
	for (int i = 0; i < 1000; i++)
	{
		unsigned short b = rng.uniform(100, 230);
		unsigned short g = rng.uniform(100, 230);
		unsigned short r = rng.uniform(100, 230);
		colors.push_back(cv::Scalar(b, g, r));
	}
	
	// Детектор для поиска характерных точек - замена на CPU версию
	// OpenCV OCL не имеет прямых аналогов для всех CUDA функций
	Ptr<FeatureDetector> detector = GFTTDetector::create(
		maxCornersConfig, qualityLevelConfig, minDistanceConfig, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
		
	Ptr<FeatureDetector> detector_small = GFTTDetector::create(
		20, qualityLevelConfig*1.5, minDistanceConfig*1.5, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
	
	// Использование CPU версии PyrLK
	TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
	Size winSizeLK(winSizeConfig, winSizeConfig);

	Mat oldFrame, oldGray, err;
	

	vector<Point2f> p0, p1, good_new;
	UMat uP0, uP1; // Используем UMat для OpenCL

	Point2f d = Point2f(0.0f, 0.0f);
	Point2f meanP0 = Point2f(0.0f, 0.0f);
	Mat T, TStab(2, 3, CV_64F), TStabInv(2, 3, CV_64F), TSearchPoints(2, 3, CV_64F);
	UMat uT, uTStab;

	vector<uchar> status;
	vector<float> errFloat;

	double tauStab = 100.0;
	double kSwitch = 0.01;
	double framePart = 0.8;

	vector <TransformParam> transforms(4);
	vector <TransformParam> movement(4);
	vector <TransformParam> movementKalman(4);
	
	for (int i = 0; i < transforms.size(); i++)
	{
		transforms[i].dx = 0.0;
		transforms[i].dy = 0.0;
		transforms[i].da = 0.0;
		movement[i].dx = 0.0;
		movement[i].dy = 0.0;
		movement[i].da = 0.0;
		movementKalman[i].dx = 0.0;
		movementKalman[i].dy = 0.0;
		movementKalman[i].da = 0.0;
	}

	// Фильтр Калмана
	int state_dim = 9;
	int meas_dim = 3;
	double FPS = 30.0;
	double dt = 1;
	double dt2 = dt*dt/2;
	
	cv::Mat A = (cv::Mat_<double>(state_dim, state_dim) <<
		1,	0,	dt,	0,	dt2,0,	0,	0,	0,
		0,	1,	0,	dt,	0,	dt2,0,	0,	0,
		0,	0,	1,	0,	dt,	0,	0,	0,	0,
		0,	0,	0,	1,	0,	dt,	0,	0,	0,
		0,	0,	0,	0,	1,	0,	0,	0,	0,
		0,	0,	0,	0,	0,	1,	0,	0,	0,
		0,	0,	0,	0,	0,	0,	1,	dt,	dt2,
		0,	0,	0,	0,	0,	0,	0,	1,	dt,
		0,	0,	0,	0,	0,	0,	0,	0,	1
		);

	cv::Mat C = (cv::Mat_<double>(meas_dim, state_dim) <<
		1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0
		);
	
	cv::Mat Q = cv::Mat::eye(state_dim, state_dim, CV_64F) * 0.00001;
	cv::Mat R = cv::Mat::eye(meas_dim, meas_dim, CV_64F) * 10000.0;
	cv::Mat P = cv::Mat::eye(state_dim, state_dim, CV_64F) * 1.0;

	KalmanFilterCV kf(dt, A, C, Q, R, P);
	cv::Mat x0 = (cv::Mat_<double>(state_dim, 1) << 0,0,0,0,0,0,0,0,0);
	kf.init(0, x0);

	bool wiener = false;
	bool threadwiener = false;
	double nsr = 0.01;
	double qWiener = 8.0;
	double LEN = 0;
	double THETA = 0.0;

	// Для счетчика кадров в секунду
	//unsigned int frameCnt = 0;
	double seconds = 0.05;
	double secondsGPUPing = 0.0;
	double secondsFullPing = 0.0;
	clock_t start = clock();
	clock_t end = clock();
	clock_t startFullPing = clock();
	clock_t endFullPing = clock();
	clock_t startGPUPing = clock();
	clock_t endGPUPing = clock();

	// Захват первого кадра
	VideoCapture capture(videoSource);
	if (cameraInUse)
	{
		if (!capture.isOpened()) {
			cerr << "Unable to connect camera!" << endl;
			return 0;
		}
		capture >> oldFrame;
	}
	else
	{
		loadImage(oldFrame, init_frame_id, filepath);
	}

	const int a = oldFrame.cols;
	const int b = oldFrame.rows;
	const double c = sqrt(a * a + b * b);
	const double atan_ba = atan2(b, a);

	// Переменные для запоминания кадров
	Mat frameShowOrig(a, b, CV_8UC3), frameOut(a, b, CV_8UC3);
	UMat uFrameStabilized(Size(a, b), CV_8UC3, USAGE_DEFAULT);

	UMat uOldFrame(Size(a, b), CV_8UC3), uFrame(Size(a, b), CV_8UC3), uFrameShowOrig(Size(a, b), CV_8UC3),
		uGray(Size(a/compressionConfig, b/compressionConfig), CV_8UC1),
		uCompressed(Size(a/compressionConfig, b/compressionConfig), CV_8UC3);

	UMat uOldGray(Size(a/compressionConfig, b/compressionConfig), CV_8UC1), 
		uOldCompressed(Size(a/compressionConfig, b/compressionConfig), CV_8UC3);
	UMat uToShow(Size(a, b), CV_8UC3);

	UMat uRoiGray;

	UMat UMatTemp_(Size(a/compressionConfig, b/compressionConfig), CV_8UC1);
	Rect roi;
	roi.x = a * ((1.0 - framePart) / 2.0);
	roi.y = b * ((1.0 - framePart) / 2.0);
	roi.width = a * framePart;
	roi.height = b * framePart;

	// Для вывода изображения на дисплей
	Mat frameStabilizatedCropResized(a, b, CV_8UC3), frame_crop;
	UMat uFrameStabilizatedCrop(roi.width, roi.height, CV_8UC3), 
		uFrameRoi(roi.width, roi.height, CV_8UC3),
		uFrameOut(Size(a, b), CV_8UC3),
		uFrameStabilizatedCropResized(Size(a, b), CV_8UC3),
		uWriterFrameToShow(outputResolution, outputResolution*b/a, CV_8UC3);

	Mat crossRef(b, a, CV_8UC3), cross(b, a, CV_8UC3);
	crossRef.setTo(colorBLACK);
	cv::rectangle(crossRef, roi, cv::Scalar(0, 10, 20), -1);
	cv::rectangle(crossRef, roi, colorGREEN, 2);
	cv::ellipse(crossRef, cv::Point2f(a/2, b/2), cv::Size(a*framePart/8, 0), 0.0, 0, 360, colorRED, 2);
	cv::ellipse(crossRef, cv::Point2f(a/2, b/2), cv::Size(0, b*framePart/8), 0.0, 0, 360, colorRED, 2);
	
	UMat uCrossRef(Size(a, b), CV_8UC3);
	crossRef.copyTo(uCrossRef);

	UMat uCross(Size(a, b), CV_8UC3);

	// Для отображения надписей
	int fontFace = FONT_HERSHEY_SIMPLEX;
	double fontScale = 0.9;
	setlocale(LC_ALL, "RU");

	vector<Point> textOrg(20), textOrgCrop(20), textOrgStab(20), textOrgOrig(20);
	if (multiScreen)
	{ 
		for (int i = 0; i < 20; i++)
		{
			textOrg[i].x = 5 + a;
			textOrg[i].y = 5 + 30*fontScale*(i+1) + b;
			textOrgCrop[i].x = 5;
			textOrgCrop[i].y = 5 + 30*fontScale*(i+1) + b;
			textOrgStab[i].x = 5;
			textOrgStab[i].y = 5 + 30*fontScale*(i+1);
			textOrgOrig[i].x = 5 + a;
			textOrgOrig[i].y = 5 + 30*fontScale*(i+1);
		}
	}
	else {
		for (int i = 0; i < 20; i++)
		{
			textOrg[i].x = 5;
			textOrg[i].y = 5 + 30*fontScale*(i+1);
		}
	}

	// Маска для нахождения точек
	Mat maskSearch = Mat::zeros(cv::Size(a/compressionConfig, b/compressionConfig), CV_8U);
	cv::rectangle(maskSearch, Rect(a*(1.0-0.9)/compressionConfig/2, b*(1.0-0.9)/compressionConfig/2, 
		a*0.9, b*0.9/compressionConfig), cv::Scalar(255), cv::FILLED);
	cv::rectangle(maskSearch, Rect(a*(1.0-0.4)/compressionConfig/2, b*(1.0-0.4)/compressionConfig/2, 
		a*0.4, b*0.4/compressionConfig), cv::Scalar(0), cv::FILLED);
	UMat uMaskSearch;
	maskSearch.copyTo(uMaskSearch);

	Mat maskSearchSmall = Mat::zeros(cv::Size(a/compressionConfig, b/compressionConfig), CV_8U);
	cv::rectangle(maskSearchSmall, Rect(a*(1.0-0.3)/compressionConfig/2, b*(1.0-0.3)/compressionConfig/2, 
		max(a,b)*0.3/compressionConfig, max(a,b)*0.3/compressionConfig), cv::Scalar(255), cv::FILLED);
	UMat uMaskSearchSmall;
	maskSearchSmall.copyTo(uMaskSearchSmall);
	UMat uMaskSearchSmallRoi;

	// Создаем объект для записи отклонения
	std::ofstream outputFile("./OutputResults/StabOutputs.txt");
	if (!outputFile.is_open())
	{
		cout << "Не удалось открыть файл для записи" << endl;
		return -1;
	}

	// Создание экземпляра класса записи видео
	VideoWriter writer, writerSmall;
	Mat writerFrame(oldFrame.rows*2, oldFrame.cols*2, CV_8UC3), 
		writerFrameSmall(oldFrame.rows, oldFrame.cols, CV_8UC3),
		writerFrameToShow;

	if (recordEnable) {
		bool isColor = (oldFrame.type() == CV_8UC3);
		int codec = VideoWriter::fourcc('a', 'v', 'c', '1');
		double fps = 30.0; 
		string filename = "./OutputVideos/TestVideo.mp4"; 
		string filenameSmall = "./OutputVideos/StabilizatedVideo.mp4";

		writer.open(filename, codec, fps, writerFrame.size(), isColor);
		if (!writer.isOpened()) {
			cerr << "Could not open the output video file for write\n";
			return -1;
		}

		writerSmall.open(filenameSmall, codec, fps, writerFrameSmall.size(), isColor);
		if (!writerSmall.isOpened()) {
			cerr << "Could not open the output video file for writeSmall\n";
			return -1;
		}
	}

	// Начало работы алгоритма
	while (true) {
		initFirstFrame(cameraInUse, capture, filepath, init_frame_id, oldFrame, uOldFrame, uOldCompressed, uOldGray, 
			uP0, p0, qualityLevelConfig, harrisKConfig, maxCornersConfig, detector, transforms, 
			kSwitch, a, b, compressionConfig, uMaskSearch, stabPossible);
		imshow("InitFirstFrame", oldFrame);
		waitKey(1);
		cout << "Initial Corners: " << p0.size() << endl;
		init_frame_id++;
		if (stabPossible)
			break;
	}
	//checkUmatFrames(uOldGray, uGray);
	for(int frameCount = init_frame_id + 1; frameCount < 450000; frameCount++){
		secondsFullPing = 0.96*secondsFullPing + 0.04*(double)(endFullPing-startFullPing)/CLOCKS_PER_SEC;
		startFullPing = clock();
		secondsGPUPing = 0.96*secondsGPUPing + 0.04*(double)(endGPUPing-startGPUPing)/CLOCKS_PER_SEC;
		if (stabPossible) {
			good_new.clear();
			for (uint i = 0; i < p1.size(); ++i)
			{
				if (status[i] && p1[i].x < (double)(a*25/32) && p1[i].x > (double)(a*6/32) && 
					p1[i].y < (double)(b*25/32) && p1[i].y > (double)(b*6/32))
				{
					good_new.push_back(p1[i]);
				}
			}
			p0.clear();
			p0 = good_new;
			
			if (p1.size() < double(maxCornersConfig*5/7) && (abs(meanP0.x-a/2) < a/6 || abs(meanP0.y-b/2) < b/6))
			{
				movementKalman[1].getTransformBoost(TSearchPoints, a, b, rng);
				cv::warpAffine(uMaskSearchSmall, uMaskSearchSmallRoi, TSearchPoints, uMaskSearchSmall.size());
				addFramePoints(uGray, p0, detector_small, uMaskSearchSmallRoi);
				removeFramePoints(p0, minDistanceConfig*0.8);
			}
			
			//uGray.copyTo(uOldGray);
			swap(uGray, uOldGray);
			convertVectorToUMat(p0, uP0);
			if (kSwitch < 0.01) kSwitch = 0.01;
			if (kSwitch < 1.0)
			{
				kSwitch *= 1.06;
				kSwitch += 0.005;
			}
			else if (kSwitch > 1.0) kSwitch = 1.0;

			if(cameraInUse) capture >> uFrame;
			else loadImage(uFrame, frameCount, filepath);
		}
		if (frameCount % 128 == 1)
		{
			end = clock();
			seconds = (double)(end-start)/CLOCKS_PER_SEC/128;
			start = clock();
		}

		if (uFrame.empty() && cameraInUse)
		{
			capture.release();
			capture = VideoCapture(videoSource);
			capture >> uFrame;
		}

		if ((multiScreen) && stabPossible) writerFrame(Rect(a,b,a,b)).setTo(colorBLACK);
		

		startGPUPing = clock();
		if (stabPossible) {
			
			cv::resize(uFrame, uCompressed, Size(a/compressionConfig, b/compressionConfig), 0.0, 0.0, INTER_AREA);
			cv::cvtColor(uCompressed, uGray, COLOR_BGR2GRAY);
		}

		if ((p0.size() < maxCornersConfig*1/5) || !stabPossible)
		{
			if (maxCornersConfig > 200) maxCornersConfig *= 0.95;
			if (p0.size() < maxCornersConfig*1/4 && stabPossible)
				detector = GFTTDetector::create(maxCornersConfig, qualityLevelConfig, minDistanceConfig, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
			
			p0.clear();
			p1.clear();

			if(cameraInUse) capture >> uFrame;
			else loadImage(uFrame, frameCount, filepath);
			
			if (!stabPossible) {
				cv::rectangle(writerFrame, Rect(a, b, a, b), cv::Scalar(0,0,0), cv::FILLED);
			}
			
			cv::resize(uFrame, uCompressed, Size(a/compressionConfig, b/compressionConfig), 0.0, 0.0, INTER_AREA);
			cv::cvtColor(uCompressed, uGray, COLOR_BGR2GRAY);

			if (frameCount % 10 == 1 && !stabPossible)
			{
				initFirstFrame(cameraInUse, capture, filepath, frameCount, oldFrame, uOldFrame, uOldCompressed, uOldGray, 
					uP0, p0, qualityLevelConfig, harrisKConfig, maxCornersConfig, detector, transforms,
					kSwitch, a, b, compressionConfig, uMaskSearch, stabPossible);
			} 
			else
			{
				initFirstFrameZero(oldFrame, uOldFrame, uOldGray, uOldCompressed, 
					uP0, p0, qualityLevelConfig, harrisKConfig, maxCornersConfig, detector, transforms, 
					kSwitch, a, b, compressionConfig, uMaskSearch, stabPossible);
			}

			if (stabPossible) {
				calcOpticalFlowPyrLK(uOldGray, uGray, p0, p1, status, errFloat, 
						winSizeLK, maxLevelConfig, termcrit, 0, 0.001);
			}
		}
		else if (stabPossible) {
			calcOpticalFlowPyrLK(uOldGray, uGray, p0, p1, status, errFloat, 
					winSizeLK, maxLevelConfig, termcrit, 0, 0.001);
		}

		if ((p1.size() > maxCornersConfig*4/5) && stabPossible) {
			maxCornersConfig *= 1.02;
			maxCornersConfig += 1;
			detector = GFTTDetector::create(maxCornersConfig, qualityLevelConfig, minDistanceConfig, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
		}
		
		if (stabPossible) {
			getBiasAndRotation(p0, p1, d, meanP0, transforms, T, compressionConfig);
			iirAdaptive(transforms, tauStab, roi, a, b, c, kSwitch, movement, movementKalman);

			kf.update((cv::Mat_<double>(3, 1) << transforms[1].dx, transforms[1].dy, transforms[1].da));
			cv::Mat state = kf.state();

			movementKalman[1].dx = state.at<double>(0, 0);
			movementKalman[1].dy = state.at<double>(1, 0);
			movementKalman[1].da = state.at<double>(6, 0);
			movementKalman[2].dx = state.at<double>(2, 0);
			movementKalman[2].dy = state.at<double>(3, 0);
			movementKalman[2].da = state.at<double>(7, 0);
			movementKalman[3].dx = state.at<double>(4, 0);
			movementKalman[3].dy = state.at<double>(5, 0);
			movementKalman[3].da = state.at<double>(8, 0);

			transforms[0].getTransform(TStab, a, b, c, atan_ba, framePart);
			transforms[0].getTransformInvert(TStabInv, a, b, c, atan_ba, framePart);
			
			cv::warpAffine(uFrame, uFrameStabilized, TStab, Size(a, b));
			uFrameStabilizatedCrop = uFrameStabilized(roi);  
			
			endGPUPing = clock();
						
			// Вывод изображения
			if (multiScreen)
			{
				cv::resize(uFrameStabilizatedCrop, uFrameStabilizatedCropResized, Size(a, b), 0.0, 0.0, INTER_NEAREST);
				uFrameStabilizatedCropResized.copyTo(frameStabilizatedCropResized);
				
				frameStabilizatedCropResized.copyTo(writerFrame(Rect(0, 0, a, b)));
				
				uFrameRoi = uFrame(roi);
				cv::resize(uFrameRoi, uFrameOut, Size(a, b), 0.0, 0.0, INTER_NEAREST);
				uFrameOut.copyTo(frameOut);
				frameOut.copyTo(writerFrame(Rect(0, b, a, b)));
				
				cv::warpAffine(uCrossRef, uCross, TStabInv, Size(a, b));
				add(uFrame, uCross, uFrameShowOrig);
				uFrameShowOrig.copyTo(frameShowOrig);
				frameShowOrig.copyTo(writerFrame(Rect(a, 0, a, b)));

				if (p0.size() > 0)
					for (uint i = 0; i < p0.size(); i++)
						circle(writerFrame, Point2f(p1[i].x*compressionConfig + a, p1[i].y*compressionConfig), 3, colors[i], -1);
								
				showServiceInfo(writerFrame, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movement, movementKalman,
					tauStab, kSwitch, framePart, p0.size(), maxCornersConfig, seconds, secondsGPUPing, secondsFullPing, 
					a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorGREEN);

				if (recordEnable)
				{
					writer.write(writerFrame);
					writerSmall.write(frameStabilizatedCropResized);
				}

				cv::resize(writerFrame, writerFrameToShow, Size(outputResolution, outputResolution*b/a), 0.0, 0.0, INTER_NEAREST);
				cv::imshow("Writed", writerFrameToShow);
			}
			if(!multiScreen) {
				cv::resize(uFrameStabilizatedCrop, uWriterFrameToShow, Size(outputResolution, outputResolution*b/a), 0.0, 0.0, INTER_NEAREST);
				uWriterFrameToShow.copyTo(writerFrameToShow);

				showServiceInfoSmall(writerFrameToShow, qWiener, nsr, wiener, threadwiener, stabPossible, 
					transforms, movementKalman, tauStab, kSwitch, framePart, p0.size(), maxCornersConfig,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab,
					fontFace, fontScale, colorGREEN);

				cv::imshow("Writed", writerFrameToShow);
			}
		}
		else {
			if (kSwitch > 0.1) kSwitch *= 0.8;
			transforms[0].dx *= 0.8;
			transforms[0].dy *= 0.8;
			transforms[0].da *= 0.8;
			transforms[0].getTransform(TStab, a, b, c, atan_ba, framePart);
			cv::warpAffine(uFrame, uFrameStabilized, TStab, Size(a, b));
			uFrameStabilizatedCrop = uFrameStabilized;
			cv::resize(uFrameStabilizatedCrop, uFrameStabilizatedCropResized, Size(a, b), 0.0, 0.0, INTER_NEAREST);
			uFrameStabilizatedCropResized.copyTo(frameStabilizatedCropResized);
			endGPUPing = clock();
			
			if (multiScreen)
			{
				cv::resize(uFrameStabilizatedCrop, uFrameStabilizatedCropResized, Size(a, b), 0.0, 0.0, INTER_NEAREST);
				uFrameStabilizatedCropResized.copyTo(frameStabilizatedCropResized);
				
				uFrameRoi = uFrame(roi);
				cv::resize(uFrameRoi, uFrameOut, Size(a, b), 0.0, 0.0, INTER_NEAREST);
				cv::warpAffine(uCrossRef, uCross, TStab, Size(a, b));
				//uFrameOut.copyTo(frameOut);
								
				uFrame.copyTo(writerFrame(Rect(a, 0, a, b)));
				uFrameOut.copyTo(writerFrame(Rect(0, 0, a, b)));
				uFrameOut.copyTo(writerFrame(Rect(0, b, a, b)));

				showServiceInfo(writerFrame, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movement, movementKalman, 
					tauStab, kSwitch, framePart, p0.size(), maxCornersConfig, seconds, secondsGPUPing, secondsFullPing, 
					a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, fontFace, fontScale, colorRED);
				
				if (recordEnable)
				{
					writer.write(writerFrame);
					writerSmall.write(frameStabilizatedCropResized);
				}
				cv::resize(writerFrame, writerFrameToShow, Size(outputResolution, outputResolution*b/a), 0.0, 0.0, INTER_NEAREST);
				cv::imshow("Writed", writerFrameToShow);
			}
			else 
			{
				cv::resize(uFrameStabilizatedCrop, uWriterFrameToShow, Size(outputResolution, outputResolution*b/a), 0.0, 0.0, INTER_NEAREST);
				uWriterFrameToShow.copyTo(writerFrameToShow);
				
				showServiceInfoSmall(writerFrameToShow, qWiener, nsr, wiener, threadwiener, stabPossible, 
					transforms, movementKalman, tauStab, kSwitch, framePart, p0.size(), maxCornersConfig,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab,
					fontFace, fontScale, colorRED);

				cv::imshow("Writed", writerFrameToShow);
			}
		}
		int keyboard = waitKey(5);
		if (keyResponse(keyboard, uFrame, frameStabilizatedCropResized, crossRef, uCrossRef, a, b, nsr, wiener, threadwiener, qWiener, tauStab, framePart, roi))
			break;
		endFullPing = clock();
		waitKey(1);
	}
	
	outputFile.close();
	capture.release();
	return 0;
}
