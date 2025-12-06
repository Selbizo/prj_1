#pragma once



#include "opencv2/core.hpp"          // Mat, Scalar, Size, Rect, Point
#include "opencv2/imgproc.hpp"       // cvtColor, rectangle, ellipse, putText
#include "opencv2/highgui.hpp"       // imshow, imwrite
#include "opencv2/calib3d.hpp"       // findChessboardCorners, calibrateCamera
#include "opencv2/videoio.hpp"       // VideoCapture

// C++
#include <vector>    
#include <iostream>  
#include <thread>    
#include <mutex>     
#include <filesystem>

#include "basicStructs.h"

using namespace cv;
using namespace std;

void createPointColors(std::vector<Scalar>& colors, cv::RNG& rng);

void convertVectorToUMat(const vector<Point2f>& p0, UMat& uP0);

int camera_calibration(int argc, char** argv);

bool keyResponse(int& keyboard, Mat& frame, Mat& croppedImg, Mat& crossRef, UMat gCrossRef,
	const double& a, const double& b, double& nsr, bool& wiener, bool& threadwiener, double& Q,
	double& tauStab, double& framePart, Rect& roi);

void showServiceInfo(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement,vector <TransformParam> movementKalman,
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color);

void showServiceInfoSmall(Mat& writerFrame, double Q, double nsr, bool wiener, bool threadwiener, bool stabPossible, vector <TransformParam> transforms, vector <TransformParam> movement, 
	double tauStab, double kSwitch, double framePart, int gP0_cols, int maxCorners,
	double seconds, double secondsPing, double secondsFullPing, int a, int b, vector <Point> textOrg, vector <Point> textOrgOrig, vector <Point> textOrgCrop, vector <Point> textOrgStab,
	int fontFace, double fontScale, Scalar color);

//#include <opencv2/opencv.hpp>

class KalmanFilterCV {
public:
	KalmanFilterCV(
		double dt,
		const cv::Mat& A,
		const cv::Mat& C,
		const cv::Mat& Q,
		const cv::Mat& R,
		const cv::Mat& P
	);

	KalmanFilterCV();

	void init();

	void init(double t0, const cv::Mat& x0);

	void update(const cv::Mat& y);

	void update(const cv::Mat& y, double dt, const cv::Mat& A);

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

// Функция для удаления пробелов в начале и конце строки
string trim(const string &str) {
    size_t start = str.find_first_not_of(" \t");
    if (start == string::npos) return "";
    size_t end = str.find_last_not_of(" \t");
    return str.substr(start, end - start + 1);
}