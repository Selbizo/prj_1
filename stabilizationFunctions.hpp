// Функции отвечающие за стабилизацию
#pragma once

// подключение необходимых модулей OpenCV
#include <opencv2/core.hpp>          
#include <opencv2/imgproc.hpp>       
#include <opencv2/videoio.hpp>       
#include <opencv2/core/cuda.hpp>     
#include <opencv2/cudaarithm.hpp>    
#include <opencv2/cudaimgproc.hpp>   
#include <opencv2/calib3d.hpp>  

#include <vector>    // std::vector
#include <iostream>  // std::cout

using namespace cv;
using namespace std;

void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame);




void initFirstFrame(VideoCapture& capture, Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldCompressed, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible)
{
	capture >> oldFrame;

	gOldFrame.upload(oldFrame);
	gOldCompressed.release();
	cuda::resize(gOldFrame, gOldCompressed, Size(a / compression, b / compression), 0.0, 0.0, cv::INTER_LINEAR);
	cuda::cvtColor(gOldCompressed, gOldGray, COLOR_BGR2GRAY);
	cuda::bilateralFilter(gOldGray, gOldGray, 3, 3.0, 1.0); //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	//cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols / frame compression , gOldGray.rows / frame compression ), 0.0, 0.0, cv::INTER_AREA);

	if (qualityLevel > 0.001 && harrisK > 0.001)
	{
		qualityLevel *= 0.6;
		harrisK *= 0.6;
	}
	else
	{
		if (maxCorners > 50)
		{
			maxCorners *= 0.98;
			d_features->setMaxCorners(maxCorners);
		}
	}
	for (int i = 0; i < 1;i++)
	{
		transforms[i].dx *= kSwitch;
		transforms[i].dy *= kSwitch;
		transforms[i].da *= kSwitch;
	}

	d_features->detect(gOldGray, gP0, mask_device);

	if ((gP0.cols > 20)) {

		p0.clear();
		gP0.download(p0);
		stab_possible = true; //true
	}
	else {
		stab_possible = false;
	}
}

void initFirstFrameZero(Mat& oldFrame, cuda::GpuMat& gOldFrame, cuda::GpuMat& gOldGray,
	cuda::GpuMat& gOldCompressed, cuda::GpuMat& gP0, vector<Point2f>& p0,
	double& qualityLevel, double& harrisK, int& maxCorners, Ptr<cuda::CornersDetector>& d_features, vector <TransformParam>& transforms,
	double& kSwitch, const int a, const int b, const int compression, cuda::GpuMat& mask_device, bool& stab_possible)
{
	gOldFrame.upload(oldFrame);
	cuda::resize(gOldFrame, gOldCompressed, Size(a / compression, b / compression), 0.0, 0.0, cv::INTER_AREA);

	cuda::cvtColor(gOldCompressed, gOldGray, COLOR_BGR2GRAY);
	cuda::bilateralFilter(gOldGray, gOldGray, 3, 3.0, 3.0);
	//cuda::resize(gOldGray, gOldGray, Size(gOldGray.cols, gOldGray.rows), 0.0, 0.0, cv::INTER_AREA);
	stab_possible = false;
}

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, Point2f& meanP0,
	vector <TransformParam>& transforms, Mat& T, const int compression)
{
	const double N = 1.0;
	for (uint i = 0; i < p1.size(); i++)
	{
		if (i == 0)
		{
			d = p1[0] - p0[0];
			meanP0 = p1[0];
		}
		d = d + (p1[i] - p0[i]);
		meanP0 = meanP0 + p1[i];
	}

	d = d * compression / (int)p0.size();
	meanP0 = meanP0 * compression / (int)p0.size();

	if (p0.empty() || p1.empty() || (p1.size() != p0.size()))
	{
		transforms[1] = TransformParam(-d.x * compression, -d.y * compression, 0.0);
		cout << "bull shit" << endl;
	}
	else
	{
		T = estimateAffine2D(p0, p1);
		transforms[1] = TransformParam(-(T.at<double>(0, 2) * N + d.x * (1.0 - N))*compression, -(T.at<double>(1, 2) * N + d.y * (1.0 - N))*compression, -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
	}
}




void addFramePoints(cuda::GpuMat& gOldGray, vector<Point2f>& p0,
	Ptr<cuda::CornersDetector>& d_features, cuda::GpuMat& gMaskSearchSmall)
{
	try {
		cuda::GpuMat gAddP0;
		vector<Point2f> addP0;

		// Check if detector and input image are valid
		if (d_features.empty() || gOldGray.empty()) {
			//cerr << "Error: Invalid detector or input image" << endl;
			return;
		}

		// Detect features
		d_features->detect(gOldGray, gAddP0, gMaskSearchSmall);

		// Download points from GPU to CPU
		if (!gAddP0.empty()) {
			gAddP0.download(addP0);
		}

		// Add points with ROI offset
		if (!addP0.empty()) {
			for (const auto& point : addP0) {
				Point2f adjustedPoint;
				adjustedPoint.x = point.x;
				adjustedPoint.y = point.y;
				p0.push_back(adjustedPoint);
			}
		}
	}
	catch (const cv::Exception& e) {
		cerr << "OpenCV exception in addFramePoints: " << e.what() << endl;
	}
	catch (const exception& e) {
		cerr << "Standard exception in addFramePoints: " << e.what() << endl;
	}
	catch (...) {
		cerr << "Unknown exception in addFramePoints" << endl;
	}
}

void addFramePoints(cuda::GpuMat& gOldGray, vector<Point2f>& p0,
	Ptr<cuda::CornersDetector>& d_features)
{
	try {
		cuda::GpuMat gAddP0;
		vector<Point2f> addP0;

		// Check if detector and input image are valid
		if (d_features.empty() || gOldGray.empty()) {
			//cerr << "Error: Invalid detector or input image" << endl;
			return;
		}

		// Detect features
		d_features->detect(gOldGray, gAddP0);

		// Download points from GPU to CPU
		if (!gAddP0.empty()) {
			gAddP0.download(addP0);
		}

		// Add points with ROI offset
		if (!addP0.empty()) {
			for (const auto& point : addP0) {
				Point2f adjustedPoint;
				adjustedPoint.x = point.x;
				adjustedPoint.y = point.y;
				p0.push_back(adjustedPoint);
			}
		}
	}
	catch (const cv::Exception& e) {
		cerr << "OpenCV exception in addFramePoints: " << e.what() << endl;
	}
	catch (const exception& e) {
		cerr << "Standard exception in addFramePoints: " << e.what() << endl;
	}
	catch (...) {
		cerr << "Unknown exception in addFramePoints" << endl;
	}
}

void removeFramePoints(vector<Point2f>& p0, double minDistance)
{
	if (p0.empty()) return;

	// Сортировка точек по оси Х
	std::sort(p0.begin(), p0.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
		return a.x < b.x;
		});

	std::vector<bool> toRemove(p0.size(), false);
	for (size_t i = 0; i < p0.size(); ++i) {
		if (toRemove[i]) continue; 

		for (size_t j = i + 1; j < p0.size(); ++j) {
			if (p0[j].x - p0[i].x > minDistance) {
				break; 
			}

			float dx = p0[j].x - p0[i].x;
			float dy = p0[j].y - p0[i].y;
			float distanceSq = dx * dx + dy * dy;

			if (distanceSq < minDistance * minDistance) {
				toRemove[j] = true;
			}
		}
	}

	for (int i = p0.size() - 1; i >= 0; --i) {
		if (toRemove[i]) {
			p0.erase(p0.begin() + i);
		}
	}
}

void iirAdaptiveOld(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, const int a, const int b, const double c, double& kSwitch)
{
	//if ((abs(transforms[0].dx) - 10.0 < 1.2 * transforms[3].dx) && (abs(transforms[0].dy) - 10.0 < 1.2 * transforms[3].dy) && (abs(transforms[0].da) - 0.02 < 1.2 * transforms[3].da))
	if ((abs(transforms[1].dx) - 20.0 < 4.0 * transforms[3].dx) && (abs(transforms[1].dy) - 20.0 < 4.0 * transforms[3].dy) && (abs(transforms[1].da) - 0.04 < 4.0 * transforms[3].da)) //проверка на выброс в данных должна устраняться фильтром Калмана
	{
		transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx);
		transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy);
		transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da);
	} 
	else 
	{
		cout<<"iirAdaptiveExeption"<<endl;
	}

	if (transforms[0].da > CV_PI)
		transforms[0].da -= CV_PI;

	if (transforms[0].da < -CV_PI)
		transforms[0].da +=CV_PI;

	if (tau_stab < 30.0)
		tau_stab *= 1.1;

	if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
		tau_stab *= 1.1;

	if (tau_stab < 500.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
	{
		tau_stab *= 1.1;
		if (tau_stab > 500.0)
			tau_stab = 500.0;
	}


	if (roi.x + (int)transforms[0].dx < 0)
	{
		transforms[0].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.x + roi.width + (int)transforms[0].dx >= a)
	{
		transforms[0].dx = (double)(a - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[0].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[0].dy >= b)
	{
		transforms[0].dy = (double)(b - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;

	//if ((abs(transforms[0].dx) - 10.0 < 2.2 * transforms[3].dx || abs(transforms[0].dx) < 0.0) && (abs(transforms[0].dy) - 10.0 < 2.2 * transforms[3].dy || abs(transforms[0].dy) < 0.0) && (abs(transforms[0].da) - 0.01 < 2.2 * transforms[3].da || abs(transforms[0].da) < 0.0))
	if (true)
	{
		transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx);
		transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy);
		transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da);
	}

	transforms[2].dx = 0.0; // - movement[1].dx; //coordinate
	transforms[2].dy = 0.0; // - movement[1].dy; //coordinate
	transforms[2].da = 0.0; // - movement[1].da; //coordinate

}


void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, const int a, const int b, const double c, double& kSwitch, vector<TransformParam>& movement, vector<TransformParam>& movementKalman)//, cv::KalmanFilter& KF)
{
	if ((abs(transforms[1].dx) - 20.0 < 3.0 * transforms[3].dx) && (abs(transforms[1].dy) - 20.0 < 3.0 * transforms[3].dy) && (abs(transforms[1].da) - 10.0*DEG_TO_RAD < 3.0 * transforms[3].da)) //проверка на выброс в данных должна устраняться фильтром Калмана
	{
		transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx) - movementKalman[1].dx;
		transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy) - movementKalman[1].dy;
		transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da) - movementKalman[1].da;
	} 
	else 
	{
		cout<<"iirAdaptiveHighPass Explosion Detected"<<endl;
	}

	if (transforms[0].da > CV_PI)
		transforms[0].da -= CV_PI;

	if (transforms[0].da < -CV_PI)
		transforms[0].da +=CV_PI;

	if (tau_stab < 30.0)
		tau_stab *= 1.2;

	if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
		tau_stab *= 1.1;

	if (tau_stab < 100.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
	{
		tau_stab *= 1.1;
		if (tau_stab > 100.0)
			tau_stab = 100.0;
	}


	if (roi.x + (int)transforms[0].dx < 0)
	{
		transforms[0].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.x + roi.width + (int)transforms[0].dx >= a)
	{
		transforms[0].dx = (double)(a - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[0].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[0].dy >= b)
	{
		transforms[0].dy = (double)(b - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;

	if (true)
	{
		transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx - movementKalman[1].dx); //абсолютная средняя ошибка
		transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy - movementKalman[1].dy); //абсолютная средняя ошибка
		transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da - movementKalman[1].da); //абсолютная средняя ошибка
	}

	double moveTau = 0.8;
	movement[1].dy = movement[1].dy*moveTau + transforms[0].dy*(1.0 - moveTau); //velocity first derivative
	movement[1].da = movement[1].da*moveTau + transforms[0].da*(1.0 - moveTau); //velocity first derivative
	movement[1].dx = movement[1].dx*moveTau + transforms[0].dx*(1.0 - moveTau); //velocity first derivative 

	movement[0].dy = movement[1].dy + movement[0].dy*0.95; //coordinate
	movement[0].da = movement[1].da + movement[0].da*0.95; //coordinate
	movement[0].dx = movement[1].dx + movement[0].dx*0.95; //coordinate

	movementKalman[0].dy = movementKalman[1].dy + movementKalman[0].dy*0.988; //coordinate
	movementKalman[0].da = movementKalman[1].da + movementKalman[0].da*0.988; //coordinate
	movementKalman[0].dx = movementKalman[1].dx + movementKalman[0].dx*0.988; //coordinate

	transforms[2].dx = 0.0; // - movement[1].dx; //coordinate
	transforms[2].dy = 0.0; // - movement[1].dy; //coordinate
	transforms[2].da = 0.0; // - movement[1].da; //coordinate

}

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, const int a, const int b, const double c, double& kSwitch, vector<TransformParam>& movement, vector<TransformParam>& movementKalman)//, cv::KalmanFilter& KF)
{
	if ((abs(transforms[1].dx) - 20.0 < 3.0 * transforms[3].dx) && (abs(transforms[1].dy) - 20.0 < 3.0 * transforms[3].dy) && (abs(transforms[1].da) - 10.0*DEG_TO_RAD < 3.0 * transforms[3].da)) //проверка на выброс в данных должна устраняться фильтром Калмана
	{
		transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx);
		transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy);
		transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da);
	} 
	else 
	{
		cout<<"iirAdaptiveHighPass Explosion Detected"<<endl;
	}

	if (transforms[0].da > CV_PI)
		transforms[0].da -= CV_PI;

	if (transforms[0].da < -CV_PI)
		transforms[0].da +=CV_PI;

	if (tau_stab < 30.0)
		tau_stab *= 1.2;

	if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
		tau_stab *= 1.1;

	if (tau_stab < 100.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
	{
		tau_stab *= 1.1;
		if (tau_stab > 100.0)
			tau_stab = 100.0;
	}


	if (roi.x + (int)transforms[0].dx < 0)
	{
		transforms[0].dx = double(1 - roi.x);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.x + roi.width + (int)transforms[0].dx >= a)
	{
		transforms[0].dx = (double)(a - roi.x - roi.width);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (roi.y + (int)transforms[1].dy < 0)
	{
		transforms[0].dy = (double)(1 - roi.y);
		if (tau_stab > 10) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}
	else if (roi.y + roi.height + (int)transforms[0].dy >= b)
	{
		transforms[0].dy = (double)(b - roi.y - roi.height);
		if (tau_stab > 50) {
			tau_stab *= 0.9;
			//transforms[0].da *= 0.999;
			kSwitch *= 0.95;
		}
	}

	if (kSwitch < 1.0)
		tau_stab *= (4.0 + kSwitch) / 5.0;

	if (true)
	{
		transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx); //абсолютная средняя ошибка
		transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy); //абсолютная средняя ошибка
		transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da); //абсолютная средняя ошибка
	}

	double moveTau = 0.8;
	movement[1].dy = movement[1].dy*moveTau + transforms[0].dy*(1.0 - moveTau); //velocity first derivative
	movement[1].da = movement[1].da*moveTau + transforms[0].da*(1.0 - moveTau); //velocity first derivative
	movement[1].dx = movement[1].dx*moveTau + transforms[0].dx*(1.0 - moveTau); //velocity first derivative 

	movement[0].dy = movement[1].dy + movement[0].dy*0.95; //coordinate
	movement[0].da = movement[1].da + movement[0].da*0.95; //coordinate
	movement[0].dx = movement[1].dx + movement[0].dx*0.95; //coordinate

	movementKalman[0].dy = movementKalman[1].dy + movementKalman[0].dy*0.988; //coordinate
	movementKalman[0].da = movementKalman[1].da + movementKalman[0].da*0.988; //coordinate
	movementKalman[0].dx = movementKalman[1].dx + movementKalman[0].dx*0.988; //coordinate

}