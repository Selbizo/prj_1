//Базовые функции
#pragma once

//OpenCV
#include <opencv2/core.hpp>          // Mat, Scalar, Size, Rect, Point
#include <opencv2/imgproc.hpp>       // cvtColor, rectangle, ellipse, putText
#include <opencv2/highgui.hpp>       // imshow, imwrite
#include <opencv2/calib3d.hpp>       // findChessboardCorners, calibrateCamera
#include <opencv2/videoio.hpp>       // VideoCapture
#include <opencv2/cudaarithm.hpp>    // GpuMat, upload, download

// C++
#include <vector>    // std::vector
#include <iostream>  // std::cout
#include <thread>    // std::thread
#include <mutex>     // std::mutex


using namespace cv;
using namespace std;


// 
const double DEG_TO_RAD = CV_PI / 180.0;
const double RAD_TO_DEG = 180.0 / CV_PI;

Scalar colorRED   (48, 62,  255);
Scalar colorYELLOW(5,  188, 251);
Scalar colorGREEN (82, 156,  23);
Scalar colorBLUE  (239,107,  23);
Scalar colorPURPLE(180,  0, 180);
Scalar colorWHITE (255,255, 255);
Scalar colorBLACK (0,    0,   0);


struct TransformParam
{
	TransformParam() {}
	TransformParam(double _dx, double _dy, double _da)
	{
		dx = _dx;
		dy = _dy;
		da = _da;
	}

	double dx;
	double dy;
	double da; // angle

	const void getTransform(Mat& T, double a, double b, double c, double atan_ba, double crop)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

	}

	const void getTransform(Mat& T)
	{
		// Reconstruct transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(da);
		T.at<double>(0, 1) = -sin(da);
		T.at<double>(1, 0) = sin(da);
		T.at<double>(1, 1) = cos(da);
		T.at<double>(0, 2) = dx;
		T.at<double>(1, 2) = dy;

	}
	
	const void getTransformInvert(Mat& T, double a, double b, double c, double atan_ba, double crop)
	{
		// Reconstruct unverted transformation matrix accordingly to new values
		T.at<double>(0, 0) = cos(-da);
		T.at<double>(0, 1) = -sin(-da);
		T.at<double>(1, 0) = sin(-da);
		T.at<double>(1, 1) = cos(-da);
		T.at<double>(0, 2) = -dx;
		T.at<double>(1, 2) = -dy;

	}

	const void getTransformBoost(Mat& T, const int a, const int b, RNG rng)
	{
		T.at<double>(0, 0) = cos(0.0);
		T.at<double>(0, 1) = -sin(0.0);
		T.at<double>(1, 0) = sin(0.0);
		T.at<double>(1, 1) = cos(0.0);
		T.at<double>(0, 2) = a*atan((dx)*0.4)/4 + rng.uniform(-10, 10);
		T.at<double>(1, 2) = b*atan((dy)*0.4)/4 + rng.uniform(-10, 10);
	}
};

static void download(const cuda::GpuMat& d_mat, vector<Point2f>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_32FC2, (void*)&vec[0]);
	d_mat.download(mat);
}

static void download(const cuda::GpuMat& d_mat, vector<uchar>& vec)
{
	vec.resize(d_mat.cols);
	Mat mat(1, d_mat.cols, CV_8UC1, (void*)&vec[0]);
	d_mat.download(mat);
}

int camera_calibration(int argc, char** argv) {

	(void)argc;
	(void)argv;

	std::vector<cv::String> fileNames;
	cv::glob("D:/CV_camera_calibration_images/*.jpg", fileNames, false);
	//cv::glob("D:/CV_camera_calibration_images/png/Image*.png", fileNames, false);
	//cv::glob("../calibration/Image*.png", fileNames, false);
	cv::Size patternSize(18 - 1, 12 - 1); //18 12
	std::vector<std::vector<cv::Point2f>> q(fileNames.size());

	std::vector<std::vector<cv::Point3f>> Q;
	// 1. Generate checkerboard (world) coordinates Q. The board has 25 x 18
	// fields with a size of 15x15mm

	int checkerBoard[2] = { 18,12 };
	// Defining the world coordinates for 3D points
	std::vector<cv::Point3f> objp;
	for (int i = 1; i < checkerBoard[1]; i++) {
		for (int j = 1; j < checkerBoard[0]; j++) {
			objp.push_back(cv::Point3f(j, i, 0));
		}
	}

	std::vector<cv::Point2f> imgPoint;
	// Detect feature points
	std::size_t i = 0;
	for (auto const& f : fileNames) {
		std::cout << std::string(f) << std::endl;

		// 2. Read in the image an call cv::findChessboardCorners()
		cv::Mat img = cv::imread(fileNames[i]);
		cv::Mat gray;

		cv::cvtColor(img, gray, cv::COLOR_RGB2GRAY);

		bool patternFound = cv::findChessboardCorners(gray, patternSize, q[i], cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);

		// 2. Use cv::cornerSubPix() to refine the found corner detections
		if (patternFound) {
			cv::cornerSubPix(gray, q[i], cv::Size(11, 11), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));
			Q.push_back(objp);
		}

		// Display
		cv::drawChessboardCorners(img, patternSize, q[i], patternFound);
		cv::imshow("chessboard detection", img);
		//cv::waitKey(0);

		i++;
	}


	cv::Matx33f K(cv::Matx33f::eye()); // intrinsic camera matrix
	cv::Vec<float, 5> k(0, 0, 0, 0, 0); // distortion coefficients

	std::vector<cv::Mat> rvecs, tvecs;
	std::vector<double> stdIntrinsics, stdExtrinsics, perViewErrors;
	int flags = cv::CALIB_FIX_ASPECT_RATIO + cv::CALIB_FIX_K3 +
		cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_FIX_PRINCIPAL_POINT;
	cv::Size frameSize(1280, 720);

	std::cout << "Calibrating..." << std::endl;
	// 4. Call "float error = cv::calibrateCamera()" with the input coordinates
	// and output parameters as declared above...

	float error = cv::calibrateCamera(Q, q, frameSize, K, k, rvecs, tvecs, flags);

	std::cout << "Reprojection error = " << error << "\nK =\n"
		<< K << "\nk=\n"
		<< k << std::endl;

	// Precompute lens correction interpolation
	cv::Mat mapX, mapY;
	cv::initUndistortRectifyMap(K, k, cv::Matx33f::eye(), K, frameSize, CV_32FC1,
		mapX, mapY);

	// Show lens corrected images
	for (auto const& f : fileNames) {
		std::cout << std::string(f) << std::endl;

		cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);

		cv::Mat imgUndistorted;
		// 5. Remap the image using the precomputed interpolation maps.
		cv::remap(img, imgUndistorted, mapX, mapY, cv::INTER_LINEAR);

		// Display
		cv::imshow("undistorted image", imgUndistorted);
		cv::waitKey(0);
	}

	cv::VideoCapture cap(0);
	cv::Mat frame, undistorted;
	while (true) {
		cap >> frame;
		if (frame.empty()) break;

		cv::remap(frame, undistorted, mapX, mapY, cv::INTER_LINEAR);


		cv::imshow("Original", frame);
		cv::imshow("Undistorted", undistorted);

		if (cv::waitKey(1) == 27) break;
			}


	return 0;
}

bool keyResponse(int& keyboard, Mat& frame, Mat& croppedImg, Mat& crossRef, cuda::GpuMat gCrossRef,
	const double& a, const double& b, double& nsr, bool& wiener, bool& threadwiener, double& Q,
	double& tauStab, double& framePart, Rect& roi)
{
	if (keyboard == 'c')
	{
		imwrite("./OutputResults/imgInCam.jpg", frame);
		imwrite("./OutputResults/imgOutCam.jpg", croppedImg);
	}
	if (keyboard == 'q' || keyboard == 27)
		return true;
	if (keyboard == '8')
	{
		nsr = nsr * 0.8;
	}
	if (keyboard == '7')
	{
		nsr = nsr * 1.25;
	}
	if (keyboard == '1')
	{
		wiener = wiener ^ true;
	}
	if (keyboard == 't')
	{
		threadwiener = threadwiener ^ true;
	}

	if (keyboard == '6')
	{
		if (Q < 20.0)
			Q = Q * 1.05;
	}
	if (keyboard == '5')
	{
		if (Q > 1.0)
			Q = Q * 0.95;
		if (Q < 1.0)
			Q = 1.0;
	}
	if (keyboard == '4')
	{
		if (tauStab < 4000)
			tauStab = tauStab * 2;

	}
	if (keyboard == '3')
	{
		if (tauStab > 4)
			tauStab = tauStab / 2;
	}
	if (keyboard == 's' || keyboard == 'S')
	{
		if (framePart < 0.95)
		{
			framePart *= 1.01;
			if (framePart > 0.9)
				framePart = 0.9;
			roi.x = a * ((1.0 - framePart) / 2.0);
			roi.y = b * ((1.0 - framePart) / 2.0);
			roi.width = a * framePart;
			roi.height = b * framePart;

			//cv::rectangle(crossRef, Rect(0, 0, a, b), Scalar(0, 0, 0), FILLED); 
			crossRef.setTo(colorBLACK);
			cv::rectangle(crossRef, roi, Scalar(0, 10, 20), -1); 
			cv::rectangle(crossRef, roi, colorGREEN, 2); //    
			cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(a * framePart / 8, 0), 0.0, 0, 360, colorRED, 2);
			cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(0, b * framePart / 8), 0.0, 0, 360, colorRED, 2);
			gCrossRef.upload(crossRef);
		}
	}
	if (keyboard == 'w' || keyboard == 'W')
	{
		if (framePart > 0.05)
		{
			framePart *= 0.99;
			if (framePart < 0.05)
				framePart = 0.05;
			roi.x = a * ((1.0 - framePart) / 2.0);
			roi.y = b * ((1.0 - framePart) / 2.0);
			roi.width = a * framePart;
			roi.height = b * framePart;

			//cv::rectangle(crossRef, Rect(0, 0, a, b), Scalar(0, 0, 0), FILLED); //    
			crossRef.setTo(colorBLACK);
			cv::rectangle(crossRef, roi, Scalar(0, 10, 20), -1); //    
			cv::rectangle(crossRef, roi, colorGREEN, 2); //    
			cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(a * framePart / 8, 0), 0.0, 0, 360, colorRED, 2);
			cv::ellipse(crossRef, cv::Point2f(a / 2, b / 2), cv::Size(0, b * framePart / 8), 0.0, 0, 360, colorRED, 2);
			gCrossRef.upload(crossRef);
		}
	}
	return false;
}


void showServiceInfo(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement,vector <TransformParam> movementKalman,
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color)
{
	unsigned short temp_i = 0;
	cv::putText(writerFrame, format("WnrFltr Q[5][6] = %2.1f, SNR[7][8] = %2.1f", Q, 1 / nsr),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Wnr On[1] %d, threads On[t] %d, stab On %d", wiener, threadwiener, stabPossible),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("transforms X %+2.1f  Y %+2.1f  Roll %+2.1f", transforms[0].dx, transforms[0].dy, transforms[0].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("vX %+2.1f vY %+2.1f vRoll %+2.1f", transforms[1].dx, transforms[1].dy, transforms[1].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("aX %+2.1f aY %+2.1f aRoll %+2.1f", transforms[2].dx, transforms[2].dy, transforms[2].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;

	cv::putText(writerFrame, format("sko vX %+2.1f vY %+2.1f vRoll %+2.1f", transforms[3].dx, transforms[3].dy, transforms[3].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	++temp_i;
	cv::putText(writerFrame, format("IIR X %+2.1f  Y %+2.1f  Roll %+2.1f", movement[0].dx, movement[0].dy, movement[0].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("vX %+2.1f vY %+2.1f vRoll %+2.1f", movement[1].dx, movement[1].dy, movement[1].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("aX %+2.1f aY %+2.1f aRoll %+2.1f", movement[2].dx, movement[2].dy, movement[2].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	++temp_i;
	cv::putText(writerFrame, format("KF X %+2.1f  Y %+2.1f  Roll %+2.1f", movementKalman[0].dx, movementKalman[0].dy, movementKalman[0].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("vX %+2.1f vY %+2.1f vRoll %+2.1f", movementKalman[1].dx, movementKalman[1].dy, movementKalman[1].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("aX %+2.1f aY %+2.1f aRoll %+2.1f", movementKalman[2].dx, movementKalman[2].dy, movementKalman[2].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("a2X %+2.1f a2Y %+2.1f a2Roll %+2.1f", movementKalman[3].dx, movementKalman[3].dy, movementKalman[3].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale , color, 2, 8, false); ++temp_i;
	++temp_i;
	cv::putText(writerFrame, format("Crop[w][s] = %+2.2f, tauStab = %2.0f , %d Current corners of %d.", 1 / framePart, tauStab, gP0_cols, maxCorners),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("FPS = %+2.1f, GPU time = %1.3f ms, Ping = %1.3f ms.", 1 / seconds, secondsPing, secondsFullPing),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Image resolution: %d x %d.", a, b),
		textOrg[temp_i], fontFace, fontScale / 1.2, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("ORIGINAL VIDEO"), textOrgOrig[0], fontFace, fontScale * 1.3, color, 2, 8, false);
	cv::putText(writerFrame, format("Stab OFF"), textOrgCrop[0], fontFace, fontScale * 1.3, colorRED, 2, 8, false);
	cv::putText(writerFrame, format("Stab ON"), textOrgStab[0], fontFace, fontScale * 1.3, colorGREEN, 2, 8, false);

}

void showServiceInfoSmall(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement, 
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color)
{
	unsigned short temp_i = 0;
	
	//cv::putText(writerFrame, format("WnrFltr Q[5][6] = %+2.1f, SNR[7][8] = %+2.1f", Q, 1 / nsr),
		//textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Wnr [1] %d, threads [t] %d, stab %d", wiener, threadwiener, stabPossible),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("[X Y Roll] %+2.1f %+2.1f %+2.1f]", transforms[2].dx, transforms[2].dy, transforms[2].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;++temp_i;
	cv::putText(writerFrame, format(" X %+2.1f  Y %+2.1f  Roll %+2.1f", movement[0].dx, movement[0].dy, movement[0].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("vX %+2.1f vY %+2.1f vRoll %+2.1f", movement[3].dx, movement[3].dy, movement[3].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("aX %+2.1f aY %+2.1f aRoll %+2.1f", movement[2].dx, movement[2].dy, movement[2].da * RAD_TO_DEG),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;++temp_i;
	cv::putText(writerFrame, format("tr_0[dX dY dRoll] %+2.2f %+2.2f %+2.2f]", transforms[0].dx, transforms[0].dy, transforms[0].da),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("tr_1[X Y Roll] %+2.2f %+2.2f %+2.2f]", transforms[1].dx, transforms[1].dy, transforms[1].da),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("tr_2[X Y Roll] %+2.2f %+2.2f %+2.2f]", transforms[2].dx, transforms[2].dy, transforms[2].da),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("[skoX skoY skoRoll] %+2.2f %+2.2f %+2.2f]", transforms[3].dx, transforms[3].dy, transforms[3].da),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	//cv::putText(writerFrame, format("[vX vY vRoll] %+2.2f %+2.2f %+2.2f]", velocity[0].dx, velocity[0].dy, velocity[0].da),
	//	textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Tau[3][4]= %3.0f frames, gain = %1.2f", tauStab, kSwitch),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Crop[w][s] = %+2.2f, %d Corners of %d.", 1 / framePart, gP0_cols, maxCorners),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("FPS= %+2.1f, GPU= %1.3f ms, Ping= %1.3f ms.", 1 / seconds, secondsPing, secondsFullPing),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;
	cv::putText(writerFrame, format("Resolution: %d x %d.", a, b),
		textOrg[temp_i], fontFace, fontScale, color, 2, 8, false); ++temp_i;

}

//#include <opencv2/opencv.hpp>

class KalmanFilterCV {
public:
	/**
	 * Create a Kalman filter with the specified matrices.
	 *   A - System dynamics matrix
	 *   C - Output matrix
	 *   Q - Process noise covariance
	 *   R - Measurement noise covariance
	 *   P - Estimate error covariance
	 */
	KalmanFilterCV(
		double dt,
		const cv::Mat& A,
		const cv::Mat& C,
		const cv::Mat& Q,
		const cv::Mat& R,
		const cv::Mat& P
	);

	/**
	 * Create a blank estimator.
	 */
	KalmanFilterCV();

	/**
	 * Initialize the filter with initial states as zero.
	 */
	void init();

	/**
	 * Initialize the filter with a guess for initial states.
	 */
	void init(double t0, const cv::Mat& x0);

	/**
	 * Update the estimated state based on measured values. The
	 * time step is assumed to remain constant.
	 */
	void update(const cv::Mat& y);

	/**
	 * Update the estimated state based on measured values,
	 * using the given time step and dynamics matrix.
	 */
	void update(const cv::Mat& y, double dt, const cv::Mat& A);

	/**
	 * Return the current state and time.
	 */
	cv::Mat state() { return x_hat; };
	double time() { return t; };

private:
	// Matrices for computation
	cv::Mat A, C, Q, R, P, K, P0;

	// System dimensions
	int m, n;

	// Initial and current time
	double t0, t;

	// Discrete time step
	double dt;

	// Is the filter initialized?
	bool initialized;

	// n-size identity
	cv::Mat I;

	// Estimated states
	cv::Mat x_hat, x_hat_new;
};

// Implementation

KalmanFilterCV::KalmanFilterCV(
	double dt,
	const cv::Mat& A,
	const cv::Mat& C,
	const cv::Mat& Q,
	const cv::Mat& R,
	const cv::Mat& P
) : A(A.clone()), C(C.clone()), Q(Q.clone()), R(R.clone()), P(P.clone()),
dt(dt), initialized(false), t0(0), t(0) {

	m = C.rows;
	n = A.rows;
	I = cv::Mat::eye(n, n, CV_64F);
	x_hat = cv::Mat::zeros(n, 1, CV_64F);
	x_hat_new = cv::Mat::zeros(n, 1, CV_64F);
	P0 = P.clone();
}

KalmanFilterCV::KalmanFilterCV() : initialized(false), t0(0), t(0), dt(0), m(0), n(0) {}

void KalmanFilterCV::init() {
	x_hat.setTo(0);
	P = P0.clone();
	t = t0;
	initialized = true;
}

void KalmanFilterCV::init(double t0, const cv::Mat& x0) {
	this->t0 = t0;
	t = t0;
	x0.copyTo(x_hat);
	P = P0.clone();
	initialized = true;
}

void KalmanFilterCV::update(const cv::Mat& y) {
	if (!initialized) {
		throw std::runtime_error("Filter is not initialized!");
	}

	// Prediction step
	x_hat_new = A * x_hat;
	P = A * P * A.t() + Q;

	// Correction step
	K = P * C.t() * (C * P * C.t() + R).inv();
	x_hat_new += K * (y - C * x_hat_new);
	P = (I - K * C) * P;

	// Update state
	x_hat_new.copyTo(x_hat);
	t += dt;
}

void KalmanFilterCV::update(const cv::Mat& y, double dt, const cv::Mat& A) {
	this->dt = dt;
	A.copyTo(this->A);
	update(y);
}