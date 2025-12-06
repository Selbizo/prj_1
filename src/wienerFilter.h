#pragma once

#include <opencv2/core.hpp>          // Mat, Scalar
#include <opencv2/imgproc.hpp>       // ellipse, normalize
#include <opencv2/highgui.hpp>       // imshow

//  C++
#include <vector>    // std::vector
#include <iostream>  // std::cout
#include <thread>	 //std::thread

//#include "basicStructs.hpp"

using namespace cv;
using namespace std;

// NO CUDA functions begin
void calcPSF(Mat& outputImg, Size filterSize, int len, double theta);

void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta);

void fftshift(const Mat& inputImg, Mat& outputImg);

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H);

void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr);

void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta);

//NO CUDA functions end