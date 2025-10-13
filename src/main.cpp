//Алгоритм стабилизации видео на основе вычисление Lucas-Kanade Optical Flow

#include "basicFunctions.h"
#include "stabilizationFunctions.h"
#include "wienerFilter.h"

#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

#define NCoef 10
#define DCgain 4

#define Ntap 31

// TransformParam iirNoise(TransformParam &NewSample,vector<TransformParam>& x, vector<TransformParam>& y) {
   
//    double FIRCoef[Ntap] = {
//          -40, -16, 28, 48, 21, -31, -56, -25, 33, 62, 29, -34, -66, -32, 34, 68, 34, -32, -66, -34, 29, 62, 33, -25, -56,-31, 21, 48, 28, -16, -40
//    };
   
//    double ACoef[NCoef+1] = {
//           12, 0, -60, 0, 120, 0, -120, 0, 60, 0, -12
//    };

//    double BCoef[NCoef+1] = {
//           64, -70, 30, -16, 29, -17, 5, -1, 1, 0, 0
//    };

//    int n;

//    //shift the old samples
//    for(n=NCoef; n>0; n--) {
//       x[n] = x[n-1];
//       y[n] = y[n-1];
//    }

//    //Calculate the new output
//    x[0] = NewSample;
//    y[0].dx = ACoef[0] * x[0].dx;
//    y[0].dy = ACoef[0] * x[0].dy;
//    y[0].da = ACoef[0] * x[0].da;

//    for (n = 1; n <= NCoef; n++)
//    {
//        y[0].dx += ACoef[n] * x[n].dx - BCoef[n] * y[n].dx;
//        y[0].dy += ACoef[n] * x[n].dy - BCoef[n] * y[n].dy;
//        y[0].da += ACoef[n] * x[n].da - BCoef[n] * y[n].da;

//    }

//    y[0].dy /= (BCoef[0]*DCgain);
//    y[0].da /= (BCoef[0]*DCgain);
//    y[0].dx /= (BCoef[0]*DCgain);

//    return y[0];
// }



int main()
{
	TransformParam noiseIn = { 0.0, 0.0, 0.0 };
	vector <TransformParam> noiseOut(2);
	for (int i = 0; i < noiseOut.size();i++)
		noiseOut[i] = {0.0, 0.0, 0.0};
	
	cv::Mat TShake(2, 3, CV_64F);

	vector <TransformParam> X(1+NCoef), Y(1 + NCoef);

	//Автоматическое создание папок
	vector <std::string> folderPath(4); 
	folderPath[0] = "./OutputVideos";
	folderPath[1] = "./OutputResults";
	folderPath[2] = "./SourceVideos";
	folderPath[3] = "./SourceVideosAuto";
    for (int tmp = 0; tmp < folderPath.size(); tmp++)
	{

		// Проверяем и создаём папку (если нужно)
		if (!fs::exists(folderPath[tmp])) {
			if (!fs::create_directory(folderPath[tmp])) {
				std::cerr << "Failed to create directory!" << std::endl;
				return -1;
			}
		}
	}
	
	// Создадим массив случайных цветов для цветов характерных точек
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 1000; i++)
	{
		unsigned short b = rng.uniform(100, 230);
		unsigned short g = rng.uniform(100, 230);
		unsigned short r = rng.uniform(100, 230);
		colors.push_back(Scalar(b, g, r));
	}
	// детектор для поиска характерных точек
	Ptr<cuda::CornersDetector > d_features = cv::cuda::createGoodFeaturesToTrackDetector(srcType,
		maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, harrisK);
		

	Ptr<cuda::CornersDetector > d_features_small = cv::cuda::createGoodFeaturesToTrackDetector(srcType,
		20, qualityLevel*1.5, minDistance*1.5, blockSize, useHarrisDetector, harrisK);
	

	Ptr<cuda::SparsePyrLKOpticalFlow> d_pyrLK_sparse = cuda::SparsePyrLKOpticalFlow::create(
		cv::Size(winSize, winSize), maxLevel, iters);

	Mat oldFrame, oldGray, err;

	vector<Point2f> p0, p1, good_new;
	cuda::GpuMat gP0, gP1;

	Point2f d = Point2f(0.0f, 0.0f);
	Point2f meanP0 = Point2f(0.0f, 0.0f);
	Mat T, TStab(2, 3, CV_64F), TStabInv(2, 3, CV_64F), TSearchPoints(2, 3, CV_64F);
	cuda::GpuMat gT, gTStab(2, 3, CV_64F);


	vector<uchar> status;
	cuda::GpuMat gStatus, gErr;
	double tauStab = 100.0;
	double kSwitch = 0.01;
	double framePart = 0.8;

	vector <TransformParam> transforms(4);
	for (int i = 0; i < transforms.size();i++)
	{
		transforms[i].dx = 0.0;
		transforms[i].dy = 0.0;
		transforms[i].da = 0.0;
	}
	vector <TransformParam> movement(4);
	
	for (int i = 0; i < movement.size();i++)
	{
		movement[i].dx = 0.0;
		movement[i].dy = 0.0;
		movement[i].da = 0.0;
	}

	vector <TransformParam> movementKalman(4);

	for (int i = 0; i < movementKalman.size();i++)
	{
		movementKalman[i].dx = 0.0;
		movementKalman[i].dy = 0.0;
		movementKalman[i].da = 0.0;

	}


	//Фильтр Калмана

		// System dimensions
	int state_dim = 9;  // vx, vy, ax, ay
	int meas_dim = 3;   // vx, vy

	// Create system matrices
	double FPS = 30.0;
	double dt = 1; //1/ FPS;
	double dt2 = dt*dt/2;
	cv::Mat A = (cv::Mat_<double>(state_dim, state_dim) <<
		1,	0,	dt,	0,	dt2,0,	0,	0,	0,	//vx	
		0,	1,	0,	dt,	0,	dt2,0,	0,	0,	//vy
		0,	0,	1,	0,	dt,	0,	0,	0,	0,	//ax
		0,	0,	0,	1,	0,	dt,	0,	0,	0,	//ay
		0,	0,	0,	0,	1,	0,	0,	0,	0,	//a2x
		0,	0,	0,	0,	0,	1,	0,	0,	0,	//a2y
		0,	0,	0,	0,	0,	0,	1,	dt,	dt2,//vroll
		0,	0,	0,	0,	0,	0,	0,	1,	dt,	//aroll
		0,	0,	0,	0,	0,	0,	0,	0,	1	//a2roll 

		);

	cv::Mat C = (cv::Mat_<double>(meas_dim, state_dim) <<
		1, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 1, 0, 0
		);

	
	cv::Mat Q = cv::Mat::eye(state_dim, state_dim, CV_64F) * 0.00001;	//low value
	cv::Mat R = cv::Mat::eye(meas_dim, meas_dim, CV_64F) * 10000.0;		//high value
	cv::Mat P = cv::Mat::eye(state_dim, state_dim, CV_64F) * 1.0;
	/**
 * Create a Kalman filter with the specified matrices.
 *   A - System dynamics matrix
 *   C - Output matrix
 *   Q - Process noise covariance
 *   R - Measurement noise covariance
 *   P - Estimate error covariance
 */
	// Create Kalman filter
	KalmanFilterCV kf(dt, A, C, Q, R, P);

	// Initialize with first measurement
	cv::Mat x0 = (cv::Mat_<double>(state_dim, 1) << 0,0,0, 0,0,0, 0,0,0);
	kf.init(0, x0);


	// переменные для фильтра Виннера
	Mat Hw, h, gray_wiener;
	cuda::GpuMat gHw, gH, gGrayWiener;

	bool wiener = false;
	bool threadwiener = false;
	double nsr = 0.01;
	double qWiener = 8.0; // скважность считывания кадра на камере (выдержка к частоте кадров) (умножена на 10)
	double LEN = 0;
	double THETA = 0.0;

	//для обработки трех каналов по Виннеру
	vector<Mat> channels(3), channelsWiener(3);
	Mat frame_wiener;

	vector<cuda::GpuMat> gChannels(3), gChannelsWiener(3);
	cuda::GpuMat gFrameWiener;

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~ для счетчика кадров в секунду ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	unsigned int frameCnt = 0;
	double seconds = 0.05;
	double secondsGPUPing = 0.0;
	double secondsFullPing = 0.0;
	clock_t start = clock();
	clock_t end = clock();

	clock_t startFullPing = clock();
	clock_t endFullPing = clock();

	clock_t startGPUPing = clock();
	clock_t endGPUPing = clock();

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Захват первого кадра ~~~~~~~~~~~~~~~~~~~~~~~~~~~//
	//~~~~~~~~~~~~~~~~~~~~~~для ситывания параметров видеопотока~~~~~~~~~~~~~~~~~//
	VideoCapture capture(videoSource);
	if (cameraInUse)
	{
		//Попытка установить 720p (1280x720)

		// capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		// capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
		// capture.set(cv::CAP_PROP_FPS, 30.0);
	
		if (!capture.isOpened()) {
			//error in opening the video input
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

	//переменные для запоминания кадров и характерных точек
	Mat frame(a, b, CV_8UC3), frameShowOrig(a, b, CV_8UC3), frameOut(a, b, CV_8UC3);
	cuda::GpuMat gFrameStabilized(a, b, CV_8UC3);

	cuda::GpuMat gFrame(a,b, CV_8UC3), gFrameShowOrig(a, b, CV_8UC3),
		gGray(a/compression, b / compression, CV_8UC1), 
		gCompressed(a / compression, b / compression, CV_8UC3);

	cuda::GpuMat gOldFrame(a, b, CV_8UC3), 
		gOldGray(a / compression, b / compression, CV_8UC1), 
		gOldCompressed(a / compression, b / compression, CV_8UC3);
	cuda::GpuMat gToShow(a, b, CV_8UC3);

	cuda::GpuMat gRoiGray;

	Rect roi;
	roi.x = a * ((1.0 - framePart) / 2.0);
	roi.y = b * ((1.0 - framePart) / 2.0);
	roi.width = a * framePart;
	roi.height = b * framePart;

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~для вывода изображения на дисплей~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat frameStabilizatedCropResized(a, b, CV_8UC3), frame_crop;
	cuda::GpuMat gFrameStabilizatedCrop(roi.width, roi.height, CV_8UC3), 
		gFrameRoi(roi.width, roi.height, CV_8UC3),
		gFrameOut(a, b, CV_8UC3),
		gFrameStabilizatedCropResized(a, b, CV_8UC3),
		gWriterFrameToShow(2000, 2000*b/a, CV_8UC3);

	Mat crossRef(b, a, CV_8UC3), cross(b, a, CV_8UC3);
	crossRef.setTo(colorBLACK); // покрасили в один цвет
	cv::rectangle(crossRef, roi, Scalar(0, 10, 20), -1); // покрасили в один цвет
	cv::rectangle(crossRef, roi, colorGREEN, 2); // покрасили в один цвет
	cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(a * framePart / 8, 0), 0.0, 0, 360, colorRED, 2);
	cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(0, b * framePart / 8), 0.0, 0, 360, colorRED, 2);
	
	cuda::GpuMat gCrossRef(a, b, CV_8UC3);
	gCrossRef.upload(crossRef);
	cuda::GpuMat gCross(a, b, CV_8UC3);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Для отображения надписей на кадре~~~~~~~~~~~~~~~~~~~~~~~~~~~
	int fontFace = FONT_HERSHEY_SIMPLEX;

	//double fontScale = 1.0*min(a,b)/1080;
	double fontScale = 1.0;
	if (writeVideo == true)
		fontScale = fontScale;

	setlocale(LC_ALL, "RU");

	vector <Point> textOrg(20);
	vector <Point> textOrgCrop(20);
	vector <Point> textOrgStab(20);
	vector <Point> textOrgOrig(20);
	if (writeVideo)
	{ 
		for (int i = 0; i < 20; i++)
		{
			textOrg[i].x = 5 + a;
			textOrg[i].y = 5 + 50* fontScale * (i + 1) + b;

			textOrgCrop[i].x = 5;
			textOrgCrop[i].y = 5 + 50 * fontScale *(i + 1) + b;

			textOrgStab[i].x = 5;
			textOrgStab[i].y = 5 + 50 * fontScale * (i + 1);

			textOrgOrig[i].x = 5 + a;
			textOrgOrig[i].y = 5 + 50 * fontScale * (i + 1);
		}
	}
	else {
		for (int i = 0; i < 20; i++)
		{
			textOrg[i].x = 5;
			textOrg[i].y = 5 + 30 * fontScale * (i + 1);
		}
	}


	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создадим маску для нахождения точек~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Mat maskSearch = Mat::zeros(cv::Size(a / compression , b / compression ), CV_8U);
	cv::rectangle(maskSearch, Rect(a * (1.0 - 0.9) / compression / 2, b * (1.0 - 0.9) / compression / 2, a * 0.9, b * 0.9 / compression ), 
		Scalar(255), FILLED); // Прямоугольная маска
	cv::rectangle(maskSearch, Rect(a * (1.0 - 0.4) / compression / 2, b * (1.0 - 0.4) / compression / 2, a * 0.4, b * 0.4 / compression),
		Scalar(0), FILLED);
	cuda::GpuMat gMaskSearch(maskSearch);

	Mat maskSearchSmall = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	cv::rectangle(maskSearchSmall, Rect(a * (1.0 - 0.3) / compression / 2, b * (1.0 - 0.3) / compression / 2, max(a,b) * 0.3 / compression, max(a,b) * 0.3 / compression),
		Scalar(255), FILLED); // Прямоугольная маска
	cuda::GpuMat gMaskSearchSmall(maskSearchSmall);
	cuda::GpuMat gMaskSearchSmallRoi;

	Mat roiMask = Mat::zeros(cv::Size(a / compression, b / compression), CV_8U);
	cv::rectangle(roiMask, Rect(a * (1.0 - 0.8) / compression / 2, b * (1.0 - 0.8) / compression / 2, a * 0.8, b * 0.8 / compression),
		Scalar(255), FILLED); // Прямоугольная маска
	//cuda::GpuMat gRoiMask(roiMask);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем GpuMat для мнимой части фильтра Винера~~~~~~~~~~~~~~~~~~~~~~~~~~~
	cuda::GpuMat zeroMatH(cv::Size(a, b), CV_32F, Scalar(0)), complexH;
	
	Ptr<cuda::DFT> forwardDFT = cuda::createDFT(cv::Size(a, b), DFT_SCALE | DFT_COMPLEX_INPUT);
	Ptr<cuda::DFT> inverseDFT = cuda::createDFT(cv::Size(a, b), DFT_INVERSE | DFT_COMPLEX_INPUT);

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем объект для записи отклонения в CSV файл~~~~~~~~~~~~~~~~~~~~~~~~~~~
	std::ofstream outputFile("./OutputResults/StabOutputs.txt");
	if (!outputFile.is_open())
	{
		cout << "Не удалось открыть файл для записи" << endl;
		return -1;
	}

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Запись заголовка в CSV файл

	// outputFile << "FrameNumber\tdx\tdy\tX\tY\ttr2x\ttr2y\ttr3x\ttr3y" << endl;
	// unsigned short temp_i = 0;
	
	//int frameCount = 0;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~СОЗДАНИЕ ЭКЗЕМПЛЯРА КЛАССА ЗАПИСИ ВИДЕО
	VideoWriter writer, writerSmall;
	cv::Mat writerFrame(oldFrame.rows * 2, oldFrame.cols * 2, CV_8UC3), writerFrameSmall(oldFrame.rows, oldFrame.cols, CV_8UC3);
	cv::Mat writerFrameToShow;

	if (writeVideo) {
		bool isColor = (oldFrame.type() == CV_8UC3);
		// int codec = VideoWriter::fourcc('D', 'I', 'V', 'X');
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

	//~~~~~~~~~~~~~~~~~~~~~~~~~~~Начало работы алгоритма~~~~~~~~~~~~~~~~~~~~~~~~//

	while (true) {
		initFirstFrame(cameraInUse, capture, filepath, init_frame_id, oldFrame, gOldFrame, gOldCompressed, gOldGray, 
			gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, 
			kSwitch, a, b, compression , gMaskSearch, stabPossible);
			init_frame_id++;
		if (stabPossible)
			break;
	}

	// while (true) {
	for(int frameCount = init_frame_id + 1; frameCount < 4500; frameCount++){
		secondsFullPing = 0.96 * secondsFullPing + 0.04 * (double)(endFullPing - startFullPing) / CLOCKS_PER_SEC;
		//++frameCount;
		startFullPing = clock();

		secondsGPUPing = 0.96 * secondsGPUPing + 0.04 * (double)(endGPUPing - startGPUPing) / CLOCKS_PER_SEC;
		if (stabPossible) {
			good_new.clear();
			for (uint i = 0; i < p1.size(); ++i)
			{
				if (status[i] && p1[i].x < (double)(a * 25 / 32) && p1[i].x > (double)(a * 6 / 32) && 
					p1[i].y < (double)(b * 25 / 32) && p1[i].y > (double)(b * 6 / 32) //remove point close to the edges
					)
				{
					good_new.push_back(p1[i]);
				}
			}
			p0.clear();
			p0 = good_new;
			
			if (p1.size() < double(maxCorners * 5 / 7) && (abs(meanP0.x - a / 2) < a / 6 || abs(meanP0.y - b / 2) < b / 6))
			{
				movementKalman[1].getTransformBoost(TSearchPoints, a, b, rng);
				//movement[1].getTransformBoost(TSearchPoints, a, b, rng);
				cuda::warpAffine(gMaskSearchSmall, gMaskSearchSmallRoi, TSearchPoints, gMaskSearchSmall.size());
				addFramePoints(gGray, p0, d_features_small, gMaskSearchSmallRoi);
				removeFramePoints(p0, minDistance*0.8);
			}

			
			gGray.copyTo(gOldGray);
			gP0.upload(p0);
			if (kSwitch < 0.01)
				kSwitch = 0.01;
			if (kSwitch < 1.0)
			{
				kSwitch *= 1.06;
				kSwitch += 0.005;

			}else if (kSwitch > 1.0)
				kSwitch = 1.0;

			//capture >> frame;

			if(cameraInUse)
			{
				capture >> frame;
			}
			else
			{
				loadImage(frame, frameCount, filepath);
			}


			noiseIn.dx = (double)(rng.uniform(-100.0, 100.0)) /4          ;// / 32 + noiseIn.dx * 31 / 32;
       		noiseIn.dy = (double)(rng.uniform(-100.0, 100.0)) /4          ;//    / 32 + noiseIn.dy * 31 / 32;
       		noiseIn.da = (double)(rng.uniform(-1000.0, 1000.0) * 0.0001)/8;// / 32 + noiseIn.da * 31 / 32;

       		noiseOut[0] = iirNoise(noiseIn, X, Y);

    		noiseOut[0].getTransform(TShake);
    		warpAffine(frame, frame, TShake, frame.size());
		}

		if (frameCnt % 128 == 1)
		{
			end = clock();
			seconds = (double)(end - start) / CLOCKS_PER_SEC / 128;
			start = clock();
		}

		if (frame.empty() && cameraInUse)
		{
			capture.release();
			capture = VideoCapture(videoSource);
			capture >> frame;
		}

		if (writeVideo && stabPossible) 
		{
			writerFrame.setTo(colorBLACK);
		}
		frameCnt++;

		startGPUPing = clock();
		if (stabPossible) {
			gFrame.upload(frame);

			cuda::resize(gFrame, gCompressed, cv::Size(a / compression , b / compression ), 0.0, 0.0, cv::INTER_AREA); //лучший метод для понижения разрешения
			cuda::cvtColor(gCompressed, gGray, COLOR_BGR2GRAY);
			cuda::bilateralFilter(gGray, gGray, 3, 1.0, 1.0); //make it adaptive to SNR
		}

		if ((gP0.cols < maxCorners * 1 / 5) || !stabPossible)
		{
			if (maxCorners > 200) //300
				maxCorners *= 0.95;
			if (gP0.cols < maxCorners * 1 / 4 && stabPossible)
				d_features->setMaxCorners(maxCorners);
			p0.clear();
			p1.clear();

			//gOldGray.release();
			
			if(cameraInUse)
			{
				capture >> frame;
			}
			else
			{
				loadImage(frame, frameCount, filepath);
			}
			
			noiseIn.dx = (double)(rng.uniform(-10.0, 10.0));
       		noiseIn.dy = (double)(rng.uniform(-10.0, 10.0));
       		noiseIn.da = (double)(rng.uniform(-0.1, 0.1));

       		noiseOut[0] = iirNoise(noiseIn, X, Y);

    		noiseOut[0].getTransform(TShake);
    		warpAffine(frame, frame, TShake, frame.size());

			if (!stabPossible) {
				cv::rectangle(writerFrame, Rect(a, b, a, b), Scalar(0, 0, 0), FILLED); // Прямоугольная маска
				//frame.copyTo(writerFrame(cv::Rect(a, 0, a, b))); //original video
			}
			gFrame.upload(frame);
			cuda::resize(gFrame, gCompressed, cv::Size(a / compression , b / compression ), 0.0, 0.0, cv::INTER_AREA); //лучший метод для понижения разрешения

			cuda::cvtColor(gCompressed, gGray, COLOR_BGR2GRAY);
			cuda::bilateralFilter(gGray, gGray, 3, 1.0, 1.0); //make it adaptive to SNR

			if (frameCnt % 10 == 1 && !stabPossible)
			{
				initFirstFrame(cameraInUse, capture, filepath, frameCount, oldFrame, gOldFrame, gOldGray, gOldCompressed, 
					gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, 
					kSwitch, a, b, compression , gMaskSearch, stabPossible); //70ms
			} 
			else
				initFirstFrameZero(oldFrame, gOldFrame, gOldGray, gOldCompressed, 
					gP0, p0, qualityLevel, harrisK, maxCorners, d_features, transforms, 
					kSwitch, a, b, compression , gMaskSearch, stabPossible); //70ms

			if (stabPossible) {
				d_pyrLK_sparse->calc(gOldGray, gCompressed, gP0, gP1, gStatus, gErr);
				gP1.download(p1);
				gErr.download(err);
			}
		}
		else if (stabPossible) {
			d_pyrLK_sparse->calc(useGray ? gOldGray : gOldFrame, useGray ? gGray : gFrame, gP0, gP1, gStatus);
			gP1.download(p1);
		}

		if ((gP1.cols > maxCorners * 4 / 5) && stabPossible) { //обновление критериев поиска точек
			{
				maxCorners *= 1.02;
				maxCorners += 1;
				d_features->setMaxCorners(maxCorners);
			}
		}
		if (stabPossible) {
			download(gStatus, status);
			getBiasAndRotation(p0, p1, d, meanP0, transforms, T, compression); //уже можно делать Винеровскую фильтрацию
			// iirAdaptive(transforms, tauStab, roi, a, b, c, kSwitch);
			//if (abs(movementKalman[1].dx)> 0.5 || abs(movementKalman[1].dy)>0.5 || abs(movementKalman[1].da)> 0.01)
				iirAdaptiveHighPass(transforms, tauStab, roi, a, b, c, kSwitch, movement, movementKalman);
			//else
			//	iirAdaptive(transforms, tauStab, roi, a, b, c, kSwitch, movement, movementKalman);

			kf.update((cv::Mat_<double>(3, 1) << transforms[1].dx, transforms[1].dy, transforms[1].da));

			cv::Mat state = kf.state();
			//std::cout << "State: " << state.t() << std::endl;
			//std::cout << "State: " << state.at<double>(0, 0) << " " << state.at<double>(2, 0) << " " << state.at<double>(4, 0) << std::endl;

			movementKalman[1].dx = state.at<double>(0, 0); //скорость
			movementKalman[1].dy = state.at<double>(1, 0); //скорость
			movementKalman[1].da = state.at<double>(6, 0); //скорость

			movementKalman[2].dx = state.at<double>(2, 0); //ускорение
			movementKalman[2].dy = state.at<double>(3, 0); //ускорение
			movementKalman[2].da = state.at<double>(7, 0); //ускорение

			movementKalman[3].dx = state.at<double>(4, 0); //вторая производная ускорения
			movementKalman[3].dy = state.at<double>(5, 0); //вторая производная ускорения
			movementKalman[3].da = state.at<double>(8, 0); //вторая производная ускорения

			transforms[0].getTransform(TStab, a, b, c, atan_ba, framePart); // получение текущего компенсирующего преобразования
			transforms[0].getTransformInvert(TStabInv, a, b, c, atan_ba, framePart); // получение текущего обратного компенсирующего преобразования для отрисовки маски

			// if (T.rows == 2 && T.cols == 3)
			// {
			// 	double xDev = T.at<double>(0, 2);
			// 	double yDev = T.at<double>(1, 2);
			// 	// Запись отклонения в CSV файл
			// 	outputFile << frameCount << "\t" << xDev << "\t" << yDev << "\t" << transforms[0].dx <<"\t" << transforms[0].dy << "\t" 
			// 		<< transforms[1].dx << "\t" << transforms[1].dy << "\t" << transforms[3].dx << "\t" << transforms[3].dy << endl;
			// }
			
			// Винеровская фильтрация
			if (wiener && kSwitch > 0.01)
			{
				LEN = sqrt(transforms[1].dx * transforms[1].dx + transforms[1].dy * transforms[1].dy) / qWiener;
				if (transforms[1].dx == 0.0)
					if (transforms[1].dy > 0.0)
						THETA = 90.0;
					else
						THETA = -90.0;
				else
					THETA = atan(transforms[1].dy / transforms[1].dx) * RAD_TO_DEG;

				cuda::bilateralFilter(gFrame, gFrame, 3, 1.0, 1.0);
				gFrame.convertTo(gFrame, CV_32F);
				cuda::split(gFrame, gChannels);

				GcalcPSF(gH, gFrame.size(), cv::Size((int)LEN * 1 + 10, (int)LEN * 1 + 10), LEN, THETA);
				GcalcWnrFilter(gH, gHw, nsr);

				// Объединяем действительную и мнимую часть фильтра в комплексную матрицу
				vector<cuda::GpuMat> planesH = { gHw, zeroMatH };

				cuda::merge(planesH, complexH);
				if (!threadwiener)
				{
					for (unsigned short i = 0; i < 3; i++) //обработка трех цветных каналов можно разделить на три потока
					{
						//gChannels[i].convertTo(gChannels[i], CV_32F);
						Gfilter2DFreqV2(gChannels[i], gChannelsWiener[i], complexH, forwardDFT, inverseDFT);
					}
				}
				else
				{
					std::thread blueChannelWiener(channelWiener, &gChannels[0], &gChannelsWiener[0], &complexH, &forwardDFT, &inverseDFT);
					std::thread greenChannelWiener(channelWiener, &gChannels[1], &gChannelsWiener[1], &complexH, &forwardDFT, &inverseDFT);
					std::thread redChannelWiener(channelWiener, &gChannels[2], &gChannelsWiener[2], &complexH, &forwardDFT, &inverseDFT);

					blueChannelWiener.join();
					greenChannelWiener.join();
					redChannelWiener.join();
				}
				cuda::merge(gChannelsWiener, gFrame);
				gFrame.convertTo(gFrame, CV_8UC3);
				cuda::bilateralFilter(gFrame, gFrame, 3, 1.0, 1.0);
			}

			cuda::warpAffine(gFrame, gFrameStabilized, TStab, cv::Size(a, b)); //8ms

			gFrameStabilizatedCrop = gFrameStabilized(roi);  
			
			endGPUPing = clock();
						
			//~~~~~~~~~~~~~~~~~~~~~~~~~~~ Вывод изображения на дисплей
			if (writeVideo)
			{
				cuda::resize(gFrameStabilizatedCrop, gFrameStabilizatedCropResized, cv::Size(a, b), 0.0, 0.0, cv::INTER_CUBIC); //8ms
				gFrameStabilizatedCropResized.download(frameStabilizatedCropResized); //9 ms
				
				frameStabilizatedCropResized.copyTo(writerFrame(cv::Rect(0, 0, a, b)));
				
				gFrameRoi = gFrame(roi); //without stab
				cuda::resize(gFrameRoi, gFrameOut, cv::Size(a, b), 0.0, 0.0, cv::INTER_NEAREST); //8ms
				gFrameOut.download(frameOut);
				frameOut.copyTo(writerFrame(cv::Rect(0, b, a, b))); //5ms
				
				cuda::warpAffine(gCrossRef, gCross, TStabInv, cv::Size(a, b));
				cuda::add(gFrame, gCross, gFrameShowOrig);
				gFrameShowOrig.download(frameShowOrig);
				frameShowOrig.copyTo(writerFrame(cv::Rect(a, 0, a, b))); //original video


				if (p0.size() > 0)
					for (uint i = 0; i < p0.size(); i++)
						circle(writerFrame, cv::Point2f(p1[i].x*compression + a, p1[i].y*compression), 3, colors[i], -1);
								
				showServiceInfo(writerFrame, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movement, movementKalman,tauStab, kSwitch, framePart, gP0.cols, maxCorners,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab, 
					fontFace, fontScale, colorGREEN);

				writer.write(writerFrame);
				writerSmall.write(frameStabilizatedCropResized);
				cv::resize(writerFrame, writerFrameToShow, cv::Size(2000, 2000*b/a), 0.0, 0.0, cv::INTER_LINEAR);
				cv::imshow("Writed", writerFrameToShow);
				//writer.write(writerFrameToShow);
			}
			if(!writeVideo) {
				cv::cuda::resize(gFrameStabilizatedCrop, gWriterFrameToShow, cv::Size(2000, 2000*b/a), 0.0, 0.0, cv::INTER_NEAREST);
				gWriterFrameToShow.download(writerFrameToShow);
				
				showServiceInfoSmall(writerFrameToShow, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movementKalman, tauStab, kSwitch, framePart, gP0.cols, maxCorners,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab,
					fontFace, fontScale, colorGREEN);

				cv::imshow("Writed", writerFrameToShow);
			}
		}
		else {
			if (kSwitch > 0.1)
				kSwitch *= 0.8;

			transforms[0].dx *= 0.8;
			transforms[0].dy *= 0.8;
			transforms[0].da *= 0.8;
			transforms[0].getTransform(TStab, a, b, c, atan_ba, framePart);
			cuda::warpAffine(gFrame, gFrameStabilized, TStab, cv::Size(a, b));
			
			gFrameStabilizatedCrop = gFrameStabilized;

			cuda::resize(gFrameStabilizatedCrop, gFrameStabilizatedCropResized, cv::Size(a, b), 0.0, 0.0, cv::INTER_NEAREST);
			gFrameStabilizatedCropResized.download(frameStabilizatedCropResized);
			//frameStabilizatedCropResized(cv::Rect(0, 0, 860, textOrg[7].y)) *= 0.3;
			//frameStabilizatedCropResized(cv::Rect(0, 0, a, textOrg[temp_i].y)) *= 0.3;
			endGPUPing = clock();
			if (writeVideo)
			{
				cuda::resize(gFrameStabilizatedCrop, gFrameStabilizatedCropResized, cv::Size(a, b), 0.0, 0.0, cv::INTER_CUBIC); //8ms
				gFrameStabilizatedCropResized.download(frameStabilizatedCropResized); //9 ms
				
				gFrameRoi = gFrame(roi);
				cuda::resize(gFrameRoi, gFrameOut, cv::Size(a, b), 0.0, 0.0, cv::INTER_NEAREST);
				cuda::warpAffine(gCrossRef, gCross, TStab, cv::Size(a, b));

				gFrameOut.download(frameOut);
								
				frame.copyTo(writerFrame(cv::Rect(a, 0, a, b))); //original video
				frameOut.copyTo(writerFrame(cv::Rect(0, 0, a, b)));
				frameOut.copyTo(writerFrame(cv::Rect(0, b, a, b)));

				showServiceInfo(writerFrame, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movement, movementKalman, tauStab, kSwitch, framePart, gP0.cols, maxCorners,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab,
					fontFace, fontScale, colorRED);
				

				writer.write(writerFrame);
				writerSmall.write(frameStabilizatedCropResized);
				cv::resize(writerFrame, writerFrameToShow, cv::Size(2000, 2000*b/a), 0.0, 0.0, cv::INTER_NEAREST);
				cv::imshow("Writed", writerFrameToShow);

			}
			else 
			{
				cv::cuda::resize(gFrameStabilizatedCrop, gWriterFrameToShow, cv::Size(2000, 2000*b/a), 0.0, 0.0, cv::INTER_NEAREST);
				gWriterFrameToShow.download(writerFrameToShow);
				
				showServiceInfoSmall(writerFrameToShow, qWiener, nsr, wiener, threadwiener, stabPossible, transforms, movementKalman, tauStab, kSwitch, framePart, gP0.cols, maxCorners,
					seconds, secondsGPUPing, secondsFullPing, a, b, textOrg, textOrgOrig, textOrgCrop, textOrgStab,
					fontFace, fontScale, colorRED);

				cv::imshow("Writed", writerFrameToShow);
			}
		}
		// Ожидание внешних команд управления с клавиатуры
		int keyboard = waitKey(80);
		if (keyResponse(keyboard, frame, frameStabilizatedCropResized, crossRef, gCrossRef, a, b, nsr, wiener, threadwiener, qWiener, tauStab, framePart, roi))
			break;
		endFullPing = clock();
	}
	outputFile.close();
	capture.release();
	return 0;
}