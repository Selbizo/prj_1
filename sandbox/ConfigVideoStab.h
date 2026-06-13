// main.cpp
#pragma once
#include <fstream>
#include <iostream>

//#include <opencv2/cudaoptflow.hpp> 
//#include <opencv2/cudawarping.hpp>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

//string videoSource = "http://192.168.0.102:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.108.144.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.139.27.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.103:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.101:4747/video"; // pixel4
//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/RoadFhd.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/ForestShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/MoveLeftRoad.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/MoveLeftRoadShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/Forestfhd.mp4"; // pad6-100, pixel4-101, pixel-102

//string videoSource = "/home/selbizo/CV/StabAndSLAM/visual_odom/src/SourceVideos/FlightShakedVideo.mp4"; // pad6-100, pixel4-101, pixel-102

// string videoSource = "http://192.168.0.104:4747/video?640x480";
string videoSource = "/home/bananapi/Opencv_projects/dataset/videos/PXL_3.mp4";
// string videoSource = "/home/bananapi/Opencv_projects/dataset/videos/PXL_3.mp4";


// int videoSource = 0;


static bool multiScreen = true;
static bool recordEnable = false;
bool stabPossible = false;

static const int compressionConfig = 2; // //4k 1->26ms 2->20ms 3->20ms

//
static const int srcType = CV_8UC1;
int maxCornersConfig = 40 / compressionConfig; //100/n
static double qualityLevelConfig = 0.005 / compressionConfig; //0.0001
static double minDistanceConfig = 4.0; //8.0
static int blockSizeConfig = 9; //45 80 
static bool useHarrisDetectorConfig = true;
double harrisKConfig = qualityLevelConfig;

// 
static const bool useGray = true;
int winSizeConfig = blockSizeConfig;
int maxLevelConfig = 5;
int itersConfig = 6;


bool cameraInUse = false;

//string videoSourceForShaked = "./SourceVideos/Forestfhd.mp4"; // pad6-100, pixel4-101, pixel-102
//int videoSourceForShaked = 0; // pad6-100, pixel4-101, pixel-102
string videoSourceForShaked = "./SourceVideos/ForestShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102

int init_frame_id = 0;
//string filepath = string("/home/bananapi/Opencv_projects/dataset/sequences/15/");
string filepath = string("/home/bananapi/Opencv_projects/dataset/videos/PXL_3/");
//cout << "Filepath: " << filepath << endl;