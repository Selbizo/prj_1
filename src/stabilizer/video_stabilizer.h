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
                 TransformParam& transformSKO, cv::Mat& stabMatrix,
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
    bool useCpuWarp{false};   // VS_WARP_CPU=1 — считать warpAffine на CPU без upload/download на GPU
    // VS_PINNED=1 — pinned оба буфера, 2 — только src (upload), 3 — только dst (download)
    int pinnedMode{0};
    // Переиспользуемые GPU-буферы (аллокация clCreateBuffer на каждом кадре стоит ~10 мс!)
    cv::UMat gpuFrame;
    cv::UMat gpuWarpDst;

    // Время обработки
    double processingTimeCapture = 0;
    double processingTimeDetectionTracking = 0;
    double processingTimeStabilization = 0;
    double processingTimeImshow = 0;
    // Под-таймеры стадии захвата (режим «только оценка смещения»): сколько
    // занимает сам cap.read() и сколько — распаковка YUYV + уменьшение до 384x216.
    double processingTimeCapRead = 0;
    double processingTimeCapPrep = 0;
    // Под-таймеры стадии стабилизации
    double processingTimeUpload = 0;
    double processingTimeWarp = 0;
    double processingTimeDownload = 0;

    // Привязка ядер
    int captureCore;
    int detectionCore;
    int stabilizationCore;
    int displayCore;

public:
    std::vector<std::thread> workers;
    std::vector<pthread_t> pthreads;
    std::atomic<bool> running;

    // Контур стабилизации (upload + warp + download) можно отключить ради пикового fps.
    // Переключается на лету клавишей B или на старте переменной VS_NO_STAB=1.
    // При выключении кадр отдаётся как есть (без компенсации движения).
    std::atomic<bool> stabilizationEnabled{true};

    // Режим «только оценка смещения» (VS_ESTIMATE_ONLY=1, CLI --estimate-only).
    // Запускаются ровно два потока — захват и детектирование/трекинг; контур
    // стабилизации и поток отображения не создаются. Кадр с камеры берётся сырым
    // (YUYV, CAP_PROP_CONVERT_RGB=0) и сразу приводится к серому 384x216, полный
    // BGR-кадр не материализуется. На выходе — dx, dy, da на кадр (строка [EST]).
    // Это самый быстрый путь: упирается в пропускную способность сенсора.
    bool estimateOnly{false};

    // Режим «максимальный fps с сырым захватом» (VS_MAX_FPS_RAW=1, CLI --max-fps-raw).
    // Обычный конвейер (окно + стабилизация), но камера читается сырым YUYV
    // (CAP_PROP_CONVERT_RGB=0), а полный BGR-кадр для окна/варпа собирается одной
    // конверсией YUYV->BGR в потоке захвата. Это убирает внутреннюю конверсию
    // OpenCV в BGR и лишний проход по кадру: захват ~10-11 мс вместо ~22-27 мс.
    // Без стабилизации (--no-stab) даёт ~100-115 fps при живом окне.
    // В estimate-only не используется (там полный BGR не нужен).
    bool maxFpsRaw{false};

    // Суммарно обработанных кадров за всё время работы в режиме estimate-only
    // (в отличие от estFrameCount, который сбрасывается каждую секунду). По нему
    // main печатает end-to-end fps: N кадров / фактическое время работы.
    std::atomic<long long> estimateFramesTotal{0};

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
