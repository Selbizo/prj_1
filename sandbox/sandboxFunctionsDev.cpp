#pragma once
//Lucas-Kanade Optical Flow
//

#include "opencv2/video/tracking.hpp"
#include <cmath>
#include <fstream>
#include <iostream>
#include <mutex>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/cudawarping.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <stdio.h>
#include <thread>
#include <vector>

using namespace cv;
using namespace std;


Mat videoStabHomograpy(cuda::GpuMat& gFrame, vector<Point2f>& p0, vector<Point2f>& p1, vector<Mat>& homoTransforms)
{
	Mat H;
	if (p0.size() >= 4) {
		H = findHomography(p0, p1, RANSAC);
		homoTransforms.push_back(H);
	}
	else {
		homoTransforms.push_back(Mat::eye(3, 3, CV_64F));
	}

	if (homoTransforms.size() < 400) {
		return homoTransforms.back();
	}
	// ��������� ������� �������������� � ����
	Mat sumH = Mat::zeros(3, 3, CV_64F);
	int count = 0;
	for (int i = homoTransforms.size() - 400; i < homoTransforms.size(); i++) {
		sumH += homoTransforms[i];
		count++;
	}
	return sumH / count;
}

void readFrameFromCapture(VideoCapture* capture, Mat* frame)
{
	*capture >> *frame;

}

// ��������� ��� �������� ����� ����������
struct OrientationAngles {
	double roll; // ���� (�������)
	double pitch; // ������ (�������)
	double yaw; // �������� (�������)
};

// ������� ��� ������ ���������� �� ����������
OrientationAngles estimateOrientationFromHomography(const cv::Mat& H, double focalLength, double cx, double cy) {
	OrientationAngles angles;

	// ������������ ������� ����������
	cv::Mat Hnorm = H / H.at<double>(2, 2);

	// ���������� ��������� ��������
	double h11 = Hnorm.at<double>(0, 0), h12 = Hnorm.at<double>(0, 1), h13 = Hnorm.at<double>(0, 2);
	double h21 = Hnorm.at<double>(1, 0), h22 = Hnorm.at<double>(1, 1), h23 = Hnorm.at<double>(1, 2);
	double h31 = Hnorm.at<double>(2, 0), h32 = Hnorm.at<double>(2, 1), h33 = Hnorm.at<double>(2, 2);

	// ���������� �����
	angles.yaw = atan2(h21, h11);
	angles.pitch = atan2(-h31, sqrt(h32 * h32 + h33 * h33));
	angles.roll = atan2(h32, h33);

	return angles;
}

// �������� ������� ���������
OrientationAngles estimateUAVOrientation(cv::cuda::GpuMat& currentFrame, cv::cuda::GpuMat& previousFrame,
	double focalLength, double cx, double cy) {
	// �������������� � grayscale
	cv::cuda::GpuMat prevGray, currGray;
	cv::cuda::cvtColor(previousFrame, prevGray, cv::COLOR_BGR2GRAY);
	cv::cuda::cvtColor(currentFrame, currGray, cv::COLOR_BGR2GRAY);

	// �������� ������������ (���������� ORB �� GPU)
	auto detector = cv::cuda::ORB::create(1000);

	// ��� �������� �������� ����� � ������������
	std::vector<cv::KeyPoint> prevKeypoints, currKeypoints;
	cv::cuda::GpuMat prevDescriptorsGPU, currDescriptorsGPU;

	// �������� � ���������� ������������
	detector->detectAndCompute(prevGray, cv::cuda::GpuMat(), prevKeypoints, prevDescriptorsGPU);
	detector->detectAndCompute(currGray, cv::cuda::GpuMat(), currKeypoints, currDescriptorsGPU);

	// ������������ ����������� � CPU ������
	cv::Mat prevDescriptors, currDescriptors;
	prevDescriptorsGPU.download(prevDescriptors);
	currDescriptorsGPU.download(currDescriptors);

	// ������������� ������������
	auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	std::vector<cv::DMatch> matches;
	matcher->match(prevDescriptors, currDescriptors, matches);

	// ���������� ������� ������������
	double minDist = DBL_MAX;
	for (const auto& m : matches) {
		if (m.distance < minDist) minDist = m.distance;
	}

	std::vector<cv::DMatch> goodMatches;
	for (const auto& m : matches) {
		if (m.distance < std::max(2.0 * minDist, 30.0)) {
			goodMatches.push_back(m);
		}
	}

	// �������� ����� ������������
	std::vector<cv::Point2f> prevPoints, currPoints;
	for (const auto& m : goodMatches) {
		prevPoints.push_back(prevKeypoints[m.queryIdx].pt);
		currPoints.push_back(currKeypoints[m.trainIdx].pt);
	}

	// ��������� ����������
	cv::Mat H;
	if (prevPoints.size() >= 4) {
		H = cv::findHomography(prevPoints, currPoints, cv::RANSAC, 3.0);
	}
	else {
		// ������������ ����� ��� ������
		return OrientationAngles{ 0, 0, 0 };
	}

	// ��������� ���� ���������� �� ����������
	return estimateOrientationFromHomography(H, focalLength, cx, cy);
}

//void addFramePoints(cuda::GpuMat& gOldGray, vector<Point2f>& p0,
//	Ptr<cuda::CornersDetector>& d_features, Rect roi)
//{
//	cuda::GpuMat gAddP0;
//	vector<Point2f> addP0;
//	d_features->detect(gOldGray, gAddP0);
//	gAddP0.download(addP0);
//	//p0.insert(p0.end(), addP0.begin(), addP0.end());
//	if (addP0.size() > 1)
//	{
//		for (uint i = 0; i < addP0.size(); i++)
//		{
//			addP0[i].x += roi.x;
//			addP0[i].y += roi.y;
//			p0.push_back(addP0[i]);
//		}
//	addP0.clear();
//	}
//}