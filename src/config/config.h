#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

// ========================= КОНСТАНТЫ =========================
constexpr bool USE_OPENCL = true;
constexpr bool USE_FP16_WARP = false;
constexpr bool DETECT_OPENCL_FP16 = false;

constexpr double DEG_TO_RAD = CV_PI / 180.0;
constexpr double RAD_TO_DEG = 180.0 / CV_PI;

// Цвета для отображения
inline const Scalar colorRED(48, 62, 255);
inline const Scalar colorYELLOW(5, 188, 251);
inline const Scalar colorGREEN(82, 156, 23);
inline const Scalar colorBLUE(239, 107, 23);
inline const Scalar colorPURPLE(180, 0, 180);
inline const Scalar colorWHITE(255, 255, 255);
inline const Scalar colorBLACK(0, 0, 0);

// Настройки системы
constexpr bool recordEnable = false;
constexpr int compressionConfig = 5;
constexpr float TAU_STAB_MAX = 120.0f;

// Настройки детектора
extern int maxCornersConfig;
extern double qualityLevelConfig;
constexpr double minDistanceConfig = 7.0;
extern int blockSizeConfig;
constexpr bool useHarrisDetectorConfig = true;
extern double harrisKConfig;

// Настройки оптического потока
constexpr int winSizeConfig = 9;
constexpr int maxLevelConfig = 5;
constexpr int itersConfig = 10;

// Память
constexpr int MAX_QUEUE_SIZE = 7;
constexpr int MAX_FRAME_BUFFER_SIZE = 10;
constexpr int SKIP_FRAMES_THRESHOLD = 3;

// Источник видео по умолчанию
extern string videoSource;

// Параметры Винеровского фильтра
extern double nsr;
extern double LEN;
extern double THETA;
extern double D;
extern double TRUE_LEN;
extern double TRUE_THETA;
extern bool wiener;
extern bool threadwiener;

// Параметры стабилизации (глобальные для доступа из модулей)
extern double framePart;
extern Rect roi;
extern int frameWidth, frameHeight;

// ========================= ГЛОБАЛЬНЫЕ OPENCL ПЕРЕМЕННЫЕ =========================
// Глобальные UMat для Винеровского фильтра (используются в stabilizer)
extern cv::UMat gHw;
extern cv::UMat gH;
extern cv::UMat gGrayWiener;
