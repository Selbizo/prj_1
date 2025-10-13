#pragma once

#include <opencv2/core.hpp>          // Mat, Scalar
#include <opencv2/imgproc.hpp>       // ellipse, normalize
#include <opencv2/highgui.hpp>       // imshow
#include <opencv2/core/cuda.hpp>     // CUDA(GpuMat)
#include <opencv2/cudaarithm.hpp>    // CUDA(sum, divide)
#include <opencv2/cudaimgproc.hpp>   // CUDA

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

//WITH CUDA functions begin

void GcalcPSF(cuda::GpuMat& outputImg, Size filterSize, Size psfSize, double len, double theta);

void GcalcPSFCircle(cuda::GpuMat& outputImg, Size filterSize, double len, double theta);

void Gfftshift(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg);

void Gfilter2DFreq(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& H);

void Gfilter2DFreqV2(const cuda::GpuMat& inputImg, cuda::GpuMat& outputImg, const cuda::GpuMat& complexH, 
	Ptr<cuda::DFT>& forwardDFT, Ptr<cuda::DFT>& inverseDFT);

void GcalcWnrFilter(const cuda::GpuMat& input_h_PSF, cuda::GpuMat& output_G, double nsr);

void Gedgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta);

void channelWiener(const cuda::GpuMat* gChannel, cuda::GpuMat* gChannelWiener,
	const cuda::GpuMat* complexH, cv::Ptr<cuda::DFT>* forwardDFT, cv::Ptr<cuda::DFT>* inverseDFT);
//WITH CUDA functions end