#pragma once
//OpenCV
#include "opencv2/core.hpp"          // Mat, Scalar, Size, Rect, Point
#include "opencv2/imgproc.hpp"       // cvtColor, rectangle, ellipse, putText
#include "opencv2/highgui.hpp"       // imshow, imwrite
#include "opencv2/calib3d.hpp"       // findChessboardCorners, calibrateCamera
#include "opencv2/videoio.hpp"       // VideoCapture

#include <opencv2/cudaarithm.hpp>    // GpuMat, upload, download

// C++
#include <vector>    
#include <iostream>  
#include <thread>    
#include <mutex>     
#include <filesystem>

#include "ConfigVideoStab.h"
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