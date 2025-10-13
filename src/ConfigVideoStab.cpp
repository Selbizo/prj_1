// main.cpp
#pragma once
#include <fstream>
#include <iostream>

#include <opencv2/cudaoptflow.hpp> 
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;



//string videoSource = "http://192.168.0.102:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.108.144.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://10.139.27.71:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.103:4747/video"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "http://192.168.0.101:4747/video"; // pixel4
//string videoSource = "./SourceVideos/RoadFhd.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/ForestShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/MoveLeftRoad.mp4"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/MoveLeftRoadShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102
//string videoSource = "./SourceVideos/Forestfhd.mp4"; // pad6-100, pixel4-101, pixel-102

string videoSource = "./SourceVideos/FlightShakedVideo.mp4"; // pad6-100, pixel4-101, pixel-102
//int videoSource = 0;

bool writeVideo = false;
bool stabPossible = false;

const int compression = 1; // //4k 1->26ms 2->20ms 3->20ms

//
int	srcType = CV_8UC1;
int maxCorners = 400 / compression; //100/n
double qualityLevel = 0.003 / compression; //0.0001
double minDistance = 6.0 / compression + 3.0; //8.0
int blockSize = 40 / compression + 8; //45 80 
bool useHarrisDetector = true;
double harrisK = qualityLevel;

// 
bool useGray = true;
int winSize = blockSize;
int maxLevel = 3 + 4/compression;
int iters = 10;


//string videoSourceForShaked = "./SourceVideos/Forestfhd.mp4"; // pad6-100, pixel4-101, pixel-102
//int videoSourceForShaked = 0; // pad6-100, pixel4-101, pixel-102
string videoSourceForShaked = "./SourceVideos/ForestShakedVideo.avi"; // pad6-100, pixel4-101, pixel-102