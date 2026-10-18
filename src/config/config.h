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
// Глубина конвейера напрямую задаёт сквозную задержку:
// лаг ≈ MAX_QUEUE_SIZE + MAX_QUEUE_SIZE + MAX_FRAME_BUFFER_SIZE кадров.
// Было 7/7/10 (до ~24 кадров ≈ 0.8 с при 30 FPS) — слишком много для live-видео.
constexpr int MAX_QUEUE_SIZE = 3;
constexpr int MAX_FRAME_BUFFER_SIZE = 4;
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

// ========================= ДИАГНОСТИКА =========================
// Подробный построчный лог кадров. Влияет на производительность (флаш stdout +
// конкуренция за мьютекс iostream), поэтому по умолчанию ВЫКЛЮЧЕН.
// Включается переменной окружения VS_VERBOSE=1.
extern bool gVerbose;

// Интерполяция при уменьшении кадра для трекинга (1920x1080 -> 384x216).
// INTER_AREA (~12 мс, лучшее качество) или INTER_LINEAR (~3 мс, быстрее).
// Переключается переменной окружения VS_FAST_RESIZE=1.
extern int captureResizeInterp;

// ========================= ГЛОБАЛЬНЫЕ OPENCL ПЕРЕМЕННЫЕ =========================
// Глобальные UMat для Винеровского фильтра (используются в stabilizer)
extern cv::UMat gHw;
extern cv::UMat gH;
extern cv::UMat gGrayWiener;
