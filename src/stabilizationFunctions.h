#pragma once
// Функции отвечающие за стабилизацию


// подключение необходимых модулей OpenCV
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>          
#include <opencv2/imgproc.hpp>       
#include <opencv2/videoio.hpp>       
#include <opencv2/core/cuda.hpp>

// #include <opencv2/cudaarithm.hpp>    
// #include <opencv2/cudaimgproc.hpp>   
// #include <opencv2/cudaoptflow.hpp> 
// #include <opencv2/cudawarping.hpp>

#include <vector>    // std::vector
#include <iostream>  // std::cout
//#include "basicFunctions.hpp"
#include "basicStructs.h"
//#include "ConfigVideoStab.hpp"

using namespace cv;
using namespace std;

void createDetectors(Ptr<FeatureDetector>& detector, Ptr<FeatureDetector>& detector_small,
    Ptr<FeatureDetector>& detector_extra);

void initFirstFrame(bool cameraInUse, VideoCapture& capture, string filepath, 
    int frame_id, Mat& oldFrame, UMat& uOldFrame, UMat& uOldCompressed, 
    UMat& uOldGray, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible);

void initFirstFrame(UMat& uOldGray, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void iir(vector<TransformParam>& transforms, double& tau_stab, Rect& roi, Mat& frame);



void initFirstFrameZero(Mat& oldFrame, UMat& uOldFrame, UMat& uOldGray,
    UMat& uOldCompressed, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible);

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, Point2f& meanP0,
	vector <TransformParam>& transforms, Mat& T, const int compression);

void addFramePoints(UMat& uOldGray, vector<Point2f>& p0,
    Ptr<FeatureDetector>& detector, UMat& uMaskSearchSmall);

void addFramePoints(UMat& uOldGray, vector<Point2f>& p0,
    Ptr<FeatureDetector>& detector);

void removeFramePoints(vector<Point2f>& p0, double minDistance);

void iirAdaptiveOld(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch);

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch, 
	vector<TransformParam>& movement, vector<TransformParam>& movementKalman);

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, 
	Rect& roi, const int a, const int b, const double c, double& kSwitch, 
	vector<TransformParam>& movement, vector<TransformParam>& movementKalman);




void loadImage(cv::Mat& image_color, int frame_id, std::string filepath);

#define NCoef 10
#define DCgain 4

#define Ntap 31

void addGaussianNoise(cv::Mat &image, double mean, double stddev);


TransformParam iirNoise(TransformParam &NewSample,vector<TransformParam>& x, vector<TransformParam>& y) {
   
   double FIRCoef[Ntap] = {
         -40, -16, 28, 48, 21, -31, -56, -25, 33, 62, 29, -34, -66, -32, 34, 68, 34, -32, -66, -34, 29, 62, 33, -25, -56,-31, 21, 48, 28, -16, -40
   };
   
   double ACoef[NCoef+1] = {
          12, 0, -60, 0, 120, 0, -120, 0, 60, 0, -12
   };

   double BCoef[NCoef+1] = {
          64, -70, 30, -16, 29, -17, 5, -1, 1, 0, 0
   };

   int n;

   //shift the old samples
   for(n=NCoef; n>0; n--) {
      x[n] = x[n-1];
      y[n] = y[n-1];
   }

   //Calculate the new output
   x[0] = NewSample;
   y[0].dx = ACoef[0] * x[0].dx;
   y[0].dy = ACoef[0] * x[0].dy;
   y[0].da = ACoef[0] * x[0].da;

   for (n = 1; n <= NCoef; n++)
   {
       y[0].dx += ACoef[n] * x[n].dx - BCoef[n] * y[n].dx;
       y[0].dy += ACoef[n] * x[n].dy - BCoef[n] * y[n].dy;
       y[0].da += ACoef[n] * x[n].da - BCoef[n] * y[n].da;

   }

   y[0].dy /= (BCoef[0]*DCgain);
   y[0].da /= (BCoef[0]*DCgain);
   y[0].dx /= (BCoef[0]*DCgain);

   return y[0];
}