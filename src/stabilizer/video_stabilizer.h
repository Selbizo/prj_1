#pragma once
#include "config/config.h"
#include "config/structures.h"
#include "sync/thread_safe_queue.h"
#include <opencv2/core/ocl.hpp>
#include <atomic>
#include <thread>
#include <condition_variable>

// ========================= IIR АДАПТИВНЫЙ ФИЛЬТР =========================
void iirAdaptive(TransformParam& transformsFirstDerivative, TransformParam& transforms,
                 TransformParam& transformSKO, cv::UMat& stabMatrix,
                 double& tauStab, cv::Rect& roi, int a, int b, double& kSwitch);

// ========================= СТАБИЛИЗАТОР =========================
class VideoStabilizer {
private:
    // Очереди конвейера
    ThreadSafeQueue<FrameData> rawFramesQueue;
    ThreadSafeQueue<FrameData> processedFramesQueue;

    // Буфер для отображения
    std::unordered_map<int, FrameData> frameBuffer;
    std::mutex bufferMutex;
    std::atomic<int> nextDisplayFrameId{0};
    std::condition_variable frameReadyCV;

    // Общие ресурсы
    std::mutex resourcesMutex;
    cv::Ptr<cv::GFTTDetector> detector;

    // Трекинг
    cv::Mat prevGrayCPU;
    std::vector<cv::Point2f> prevPoints;
    bool firstFrameForTracking = true;

    // Параметры стабилизации
    double tauStab;
    double kSwitch;
    TransformParam oldTransform;
    cv::Rect roi;
    cv::Size frameSize;
    int a, b;

    // Статистика
    std::atomic<int> fps;
    std::atomic<int> trackedPoints;
    std::atomic<int> framesProcessed;
    std::atomic<int> framesSkipped;
    std::atomic<bool> debugMode;

    // Производительность
    std::atomic<int> processingLag{0};
    const int MAX_PROCESSING_LAG = 3;
    const int MAX_BUFFER_SIZE = MAX_FRAME_BUFFER_SIZE;

    // GPU оптимизация
    bool useFP16Warp{false};

    // Время обработки
    double processingTimeCapture = 0;
    double processingTimeDetectionTracking = 0;
    double processingTimeStabilization = 0;
    double processingTimeImshow = 0;

    // Привязка ядер
    int captureCore;
    int detectionCore;
    int stabilizationCore;
    int displayCore;

public:
    std::vector<std::thread> workers;
    std::vector<pthread_t> pthreads;
    std::atomic<bool> running;

    VideoStabilizer();
    ~VideoStabilizer();

    void setUseFP16(bool value);
    void start(bool useCamera, const std::string& imageFolderPath = "");
    void stop();

private:
    // Обёртки для аффинности
    void captureThreadWrapper(bool useCamera, const std::string& imageFolderPath);
    void detectionAndTrackingThreadWrapper();
    void stabilizationThreadWrapper();
    void displayThreadWrapper();

    // Потоки
    void captureThread(bool useCamera, const std::string& imageFolderPath);
    void detectionAndTrackingThread();
    void stabilizationThread();
    void displayThread();

    // Вспомогательные
    void loadImage(cv::Mat& image, int frame_id, const std::string& filepath);
    void removeFramePoints(std::vector<cv::Point2f>& points, double minDistance);
};
