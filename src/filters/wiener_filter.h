#pragma once
#include "config/config.h"
#include <opencv2/core/ocl.hpp>

// ========================= ФИЛЬТРЫ ВИНЕРА (OpenCL) =========================

// Расчёт PSF (Point Spread Function) для размытия
void GcalcPSF(cv::UMat& outputImg, cv::Size filterSize, cv::Size psfSize, double len, double theta);

// Сдвиг центра спектра (zero-frequency component to center)
void Gfftshift(const cv::UMat& inputImg, cv::UMat& outputImg);

// 2D фильтрация в частотной области
void Gfilter2DFreq(const cv::UMat& inputImg, cv::UMat& outputImg, const cv::UMat& complexH);

// Расчёт фильтра Винера
void GcalcWnrFilter(const cv::UMat& input_h_PSF, cv::UMat& output_G, double nsr);

// Тандемное окно (taper) для краёв
void Gedgetaper(const cv::Mat& inputImg, cv::Mat& outputImg, double gamma, double beta);

// Канальная Винеровская фильтрация
void channelWiener(const cv::UMat* gChannel, cv::UMat* gChannelWiener, const cv::UMat* complexH);
