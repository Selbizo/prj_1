#pragma once
#include "config/config.h"
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <unistd.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <vector>
#include <string>

// ========================= УТИЛИТЫ ПУТИ =========================
string normalizePath(const string& path);

// ========================= УТИЛИТЫ ПОТОКОВ =========================
void setThreadAffinity(pthread_t thread, int cpu_core);
void setCurrentThreadAffinity(int cpu_core);
int getAvailableCores();

// ========================= УТИЛИТЫ КАРТИНКИ =========================
void loadImage(cv::Mat& image, int frame_id, const string& filepath);

// ========================= УТИЛИТЫ ТОЧЕК =========================
void removeFramePoints(vector<cv::Point2f>& points, double minDistance);

// ========================= УТИЛИТЫ ВИНЕРА =========================
void GcalcPSF(cv::UMat& outputImg, cv::Size filterSize, cv::Size psfSize, double len, double theta);
void Gfftshift(const cv::UMat& inputImg, cv::UMat& outputImg);
void Gfilter2DFreq(const cv::UMat& inputImg, cv::UMat& outputImg, const cv::UMat& complexH);
void GcalcWnrFilter(const cv::UMat& input_h_PSF, cv::UMat& output_G, double nsr);
void Gedgetaper(const cv::Mat& inputImg, cv::Mat& outputImg, double gamma, double beta);
void channelWiener(const cv::UMat* gChannel, cv::UMat* gChannelWiener, const cv::UMat* complexH);
