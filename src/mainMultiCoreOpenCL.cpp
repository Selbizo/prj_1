#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <deque>
#include <unordered_map>

#include <sys/stat.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

// Включаем оптимизированный warpAffine для ARM NEON
#include "warpAffine_neon_optimized.hpp"

// Включаем OpenMP для параллелизма
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace cv;
using namespace std;
namespace fs = filesystem;

// ========================= КОНСТАНТЫ И КОНФИГУРАЦИЯ =========================
const bool USE_OPENCL = true;
const bool USE_FP16_WARP = false;  // Отключаем FP16 в пользу NEON оптимизации
const bool DETECT_OPENCL_FP16 = false;  // Отключаем автодетекцию FP16
const bool USE_NEON_WARP_AFFINE = true;  // Включаем NEON оптимизацию warpAffine

const double DEG_TO_RAD = CV_PI / 180.0;
const double RAD_TO_DEG = 180.0 / CV_PI;

// Цвета для отображения
const Scalar colorRED(48, 62, 255);
const Scalar colorYELLOW(5, 188, 251);
const Scalar colorGREEN(82, 156, 23);
const Scalar colorBLUE(239, 107, 23);
const Scalar colorPURPLE(180, 0, 180);
const Scalar colorWHITE(255, 255, 255);
const Scalar colorBLACK(0, 0, 0);

// Настройки системы
const bool recordEnable = false;
const int compressionConfig = 5; // Сжатие для обработки
const float TAU_STAB_MAX = 120;
// Настройки детектора
int maxCornersConfig = 200;
double qualityLevelConfig = 0.005;
const double minDistanceConfig = 7.0;
int blockSizeConfig = 9;
const bool useHarrisDetectorConfig = true;
double harrisKConfig = 0.005;

// Настройки оптического потока
const int winSizeConfig = blockSizeConfig;
const int maxLevelConfig = 5;
const int itersConfig = 10;

// переменные для фильтра Виннера
Mat Hw, h, gray_wiener;
cv::UMat gHw, gH, gGrayWiener;

bool wiener = true;
bool threadwiener = false;
double nsr = 0.01;
double LEN = 0;
double THETA = 0.0;
double D = 0.2;

double TRUE_LEN = 0;
double TRUE_THETA = 0.0;
//для обработки трех каналов по Виннеру
vector<Mat> channels(3), channelsWiener(3);
Mat frame_wiener;

vector<cv::UMat> gChannels(3), gChannelsWiener(3);
cv::UMat gFrameWiener;



// Источник видео
string videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";

// ========================= НОВЫЕ КОНСТАНТЫ ДЛЯ УПРАВЛЕНИЯ ПАМЯТЬЮ =========================
const int MAX_QUEUE_SIZE = 7;           // Максимальный размер очередей
const int MAX_FRAME_BUFFER_SIZE = 10;    // Максимальный размер буфера кадров для отображения
const int SKIP_FRAMES_THRESHOLD = 3;    // Сколько кадров пропускать при переполнении

// ========================= СТРУКТУРЫ ДАННЫХ =========================

struct TransformParam {
    double dx;
    double dy;
    double da; // угол в радианах
    
    TransformParam() : dx(0), dy(0), da(0) {}
    TransformParam(double _dx, double _dy, double _da) 
        : dx(_dx), dy(_dy), da(_da) {}
    
    void getTransform(UMat& T) const {
        T = UMat::zeros(2, 3, CV_64F);
        Mat T_cpu = T.getMat(ACCESS_WRITE);
        T_cpu.at<double>(0, 0) = cos(da);
        T_cpu.at<double>(0, 1) = -sin(da);
        T_cpu.at<double>(0, 2) = dx;
        T_cpu.at<double>(1, 0) = sin(da);
        T_cpu.at<double>(1, 1) = cos(da);
        T_cpu.at<double>(1, 2) = dy;
    }
    
    void getTransformInvert(UMat& T) const {
        T = UMat::zeros(2, 3, CV_64F);
        Mat T_cpu = T.getMat(ACCESS_WRITE);
        T_cpu.at<double>(0, 0) = cos(da);
        T_cpu.at<double>(0, 1) = sin(da);
        T_cpu.at<double>(0, 2) = -dx;
        T_cpu.at<double>(1, 0) = -sin(da);
        T_cpu.at<double>(1, 1) = cos(da);
        T_cpu.at<double>(1, 2) = -dy;
    }
    
    void print() const {
        cout << "TransformPrint: dx=" << dx << " dy=" << dy << " da=" << da * RAD_TO_DEG << " deg" << endl;
    }
    string printString() const {
        string str = "Transform: dx= " + to_string(dx) + 
                                 " dy= " + to_string(dy) + 
                                 " da= " + to_string(da) +  "\n";
        return str; 
    }
};

struct FrameData {
    // CPU данные (используются в детекции и трекинге)
    Mat frameCPU;           // Полноразмерный цветной кадр на CPU
    Mat grayCPU;            // Сжатое серое изображение на CPU для обработки
    vector<Point2f> points; // Точки для отслеживания
    
    // GPU данные (используются только в стабилизации и отображении)
    UMat frameGPU;          // Полноразмерный кадр на GPU для warpAffine
    UMat grayGPU;           // Для совместимости, но не используется активно
    
    TransformParam transformSKO;
    TransformParam transformFirstDerivative;
    TransformParam transform;
    UMat stabMatrix;
    int frameId;
    double timestamp;
    bool shouldSkip;        // Флаг для пропуска кадра при переполнении
    
    FrameData() : frameId(0), timestamp(0), shouldSkip(false) {}
};

// ========================= ИСПРАВЛЕННАЯ ОЧЕРЕДЬ С ОГРАНИЧЕНИЕМ =========================
class ThreadSafeQueue {
private:
    deque<FrameData> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
    const size_t maxSize_;
    
public:
    ThreadSafeQueue(size_t maxSize = MAX_QUEUE_SIZE) : maxSize_(maxSize) {}
    
    bool push(FrameData data) {
        lock_guard<mutex> lock(mutex_);
        
        // Если очередь переполнена, пропускаем кадр
        if (queue_.size() >= maxSize_) {
            return false; // Кадр не добавлен
        }
        
        queue_.push_back(move(data));
        cond_.notify_one();
        return true;
    }
    
    bool try_pop(FrameData& data) {
        lock_guard<mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = move(queue_.front());
        queue_.pop_front();
        return true;
    }
    
    bool wait_and_pop(FrameData& data) {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        data = move(queue_.front());
        queue_.pop_front();
        return true;
    }
    
    bool empty() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.empty();
    }
    
    size_t size() const {
        lock_guard<mutex> lock(mutex_);
        return queue_.size();
    }
    
    void clear() {
        lock_guard<mutex> lock(mutex_);
        queue_.clear();
    }
};

// ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================

void setThreadAffinity(pthread_t thread, int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    int rc = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        cerr << "Error setting thread affinity to core " << cpu_core 
             << ": " << strerror(rc) << endl;
    } else {
        cout << "Thread bound to CPU core " << cpu_core << endl;
    }
}

void setCurrentThreadAffinity(int cpu_core) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_core, &cpuset);
    
    pthread_t current_thread = pthread_self();
    int rc = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        cerr << "Error setting current thread affinity to core " << cpu_core 
             << ": " << strerror(rc) << endl;
    } else {
        cout << "Current thread bound to CPU core " << cpu_core << endl;
    }
}

int getAvailableCores() {
    return sysconf(_SC_NPROCESSORS_ONLN);
}

// ========================= GPU ОПТИМИЗАЦИЯ (FP16 для Mali G52) =========================

struct GPUCapabilities {
    bool supportsFP16;
    bool supportsHalfType;
    string gpuName;
    
    GPUCapabilities() : supportsFP16(false), supportsHalfType(false) {}
};

// Проверить возможности GPU для FP16
GPUCapabilities detectGPUCapabilities() {
    GPUCapabilities caps;
    
    if (!ocl::haveOpenCL()) {
        cerr << "[GPU] OpenCL не доступен" << endl;
        return caps;
    }
    
    ocl::Context context = ocl::Context::getDefault();
    if (!context.ptr()) {
        cerr << "[GPU] Не удалось получить OpenCL контекст" << endl;
        return caps;
    }
    
    const ocl::Device& device = ocl::Device::getDefault();
    caps.gpuName = device.name();
    cout << "[GPU] Найдено устройство: " << caps.gpuName << endl;
    
    // Проверить расширения для FP16
    string extensions = device.extensions();
    caps.supportsFP16 = (extensions.find("cl_khr_fp16") != string::npos);
    
    if (caps.supportsFP16) {
        cout << "[GPU] ✓ FP16 поддерживается (cl_khr_fp16)" << endl;
    } else {
        cout << "[GPU] ✗ FP16 расширение не найдено" << endl;
    }
    
    // ARM Mali может иметь встроенную поддержку half-типов
    if (caps.gpuName.find("Mali") != string::npos) {
        caps.supportsHalfType = true;
        cout << "[GPU] ✓ Обнаружена Mali GPU - активирована FP16 поддержка" << endl;
    }
    
    return caps;
}

// Быстрая warpAffine с NEON оптимизацией для Banana Pi (ARM Cortex-A53)
void warpAffineOptimized(InputArray src, OutputArray dst, InputArray M, Size dsize,
                         int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar(), bool useFP16 = false) {
    
    // Используем NEON оптимизацию для основного пути
    if (USE_NEON_WARP_AFFINE && src.type() == CV_8UC3 && dsize.width >= 256 && dsize.height >= 256) {
        try {
            // Для CPU обработки - используем NEON с tile-based processing
            WarpAffineNeonOptimized::warpAffine(src, dst, M, dsize, flags, borderMode, borderValue, true);
            return;
        } catch (const exception& e) {
            cerr << "[NEON] Error in warpAffine optimization: " << e.what() << endl;
            // Fallback на стандартную версию
        }
    }
    
    // Стандартная версия для других случаев
    warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
}

// ========================= ОСНОВНЫЕ ФУНКЦИИ =========================

class VideoStabilizer {
private:
    // Очереди конвейерной обработки с ограниченным размером
    ThreadSafeQueue rawFramesQueue{MAX_QUEUE_SIZE};          // Захват -> Детекция/Трекинг
    ThreadSafeQueue processedFramesQueue{MAX_QUEUE_SIZE};    // Детекция/Трекинг -> Стабилизация
    
    // Буфер для сохранения порядка кадров при отображении
    unordered_map<int, FrameData> frameBuffer;
    mutex bufferMutex;
    atomic<int> nextDisplayFrameId{0};
    condition_variable frameReadyCV;
    
    // Общие ресурсы
    mutex resourcesMutex;
    Ptr<GFTTDetector> detector;

    // Для отслеживания точек между кадрами (все на CPU)
    Mat prevGrayCPU;           // Предыдущий серый кадр на CPU
    vector<Point2f> prevPoints;
    bool firstFrameForTracking = true;
    
    // Параметры стабилизации
    double tauStab;
    double kSwitch;
    double framePart;
    TransformParam oldTransform;
    Rect roi;
    Size frameSize;
    int a, b; // Ширина и высота кадра
    
    // Статистика
    atomic<int> fps;
    atomic<int> trackedPoints;
    atomic<int> framesProcessed;
    atomic<int> framesSkipped;  // Счетчик пропущенных кадров
    atomic<bool> debugMode;
    
    // Оптимизация производительности
    atomic<int> processingLag{0};
    const int MAX_PROCESSING_LAG = 3;  // Уменьшил с 6 до 3
    const int MAX_BUFFER_SIZE = MAX_FRAME_BUFFER_SIZE;
    
    // GPU оптимизация
    bool useFP16Warp{false};  // Флаг для использования FP16 в warpAffine
    
    double processingTimeCapture;
    double processingTimeDetectionTracking;
    double processingTimeStabilization;
    double processingTimeImshow;

    // Привязка ядер для потоков
    int captureCore;
    int detectionCore;
    int stabilizationCore;
    int displayCore;

public:
    vector<thread> workers;
    vector<pthread_t> pthreads;
    atomic<bool> running;

    VideoStabilizer() 
        : running(false), 
          tauStab(TAU_STAB_MAX/4), 
          kSwitch(0.1), 
          framePart(0.7),
          trackedPoints(0),
          framesProcessed(0),
          framesSkipped(0),
          debugMode(false),
          captureCore(-1),
          detectionCore(-1),
          stabilizationCore(-1),
          displayCore(-1) {
        
        int totalCores = getAvailableCores();
        cout << "Available CPU cores: " << totalCores << endl;
        
        if (totalCores >= 6) {
            captureCore = -1;
            detectionCore = -1;
            stabilizationCore = -1;
            displayCore = -1;
        } else if (totalCores >= 4) {
            captureCore = 0;
            detectionCore = 1;
            stabilizationCore = 2;
            displayCore = 3;
        }
        
        detector = GFTTDetector::create(
            maxCornersConfig, 
            qualityLevelConfig, 
            minDistanceConfig, 
            blockSizeConfig, 
            useHarrisDetectorConfig, 
            harrisKConfig
        );

        cout << "Detector created with maxCorners=" << maxCornersConfig << endl;
        cout << "Memory limits: Queue size=" << MAX_QUEUE_SIZE 
             << ", Frame buffer=" << MAX_FRAME_BUFFER_SIZE << endl;
    }
    
    ~VideoStabilizer() {
        stop();
    }
    
    void setUseFP16(bool value) {
        useFP16Warp = value;
        if (USE_NEON_WARP_AFFINE) {
            cout << "[NEON] NEON tile-based warpAffine оптимизация активирована для ARM Cortex-A53" << endl;
            #ifdef _OPENMP
            cout << "[OpenMP] Параллелизм включен на " << omp_get_num_procs() << " ядрах" << endl;
            #endif
        } else if (value) {
            cout << "[GPU] FP16 оптимизация активирована для warpAffine()" << endl;
        }
    }
    
    void start(bool useCamera, const string& imageFolderPath = "") {
        running = true;
        
        workers.emplace_back(&VideoStabilizer::captureThreadWrapper, this, useCamera, imageFolderPath);
        workers.emplace_back(&VideoStabilizer::detectionAndTrackingThreadWrapper, this);
        workers.emplace_back(&VideoStabilizer::stabilizationThreadWrapper, this);
        workers.emplace_back(&VideoStabilizer::displayThreadWrapper, this);
        
        for (auto& worker : workers) {
            pthreads.push_back(worker.native_handle());
        }
        
        if (captureCore >= 0) setThreadAffinity(pthreads[0], captureCore);
        if (detectionCore >= 0) setThreadAffinity(pthreads[1], detectionCore);
        if (stabilizationCore >= 0) setThreadAffinity(pthreads[2], stabilizationCore);
        if (displayCore >= 0) setThreadAffinity(pthreads[3], displayCore);
        
        cout << "Video stabilizer started with " << workers.size() << " threads" << endl;
    }
    
    void stop() {
        running = false;
        
        // Очищаем очереди при остановке
        rawFramesQueue.clear();
        processedFramesQueue.clear();
        
        {
            lock_guard<mutex> lock(bufferMutex);
            frameBuffer.clear();
            frameReadyCV.notify_all();
        }
        
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
        workers.clear();
        pthreads.clear();
        cout << "Video stabilizer stopped. Total frames processed: " << framesProcessed 
             << ", Skipped: " << framesSkipped << endl;
    }
    
private:
    void captureThreadWrapper(bool useCamera, const string& imageFolderPath) {
        if (captureCore >= 0) setCurrentThreadAffinity(captureCore);
        captureThread(useCamera, imageFolderPath);
    }
    
    void detectionAndTrackingThreadWrapper() {
        if (detectionCore >= 0) setCurrentThreadAffinity(detectionCore);
        detectionAndTrackingThread();
    }
    
    void stabilizationThreadWrapper() {
        if (stabilizationCore >= 0) setCurrentThreadAffinity(stabilizationCore);
        stabilizationThread();
    }
    
    void displayThreadWrapper() {
        if (displayCore >= 0) setCurrentThreadAffinity(displayCore);
        displayThread();
    }
    
    // ========================= ПОТОК ЗАХВАТА КАДРОВ (С ЗАМЕДЛЕНИЕМ ДЛЯ ВИДЕОФАЙЛОВ) =========================
    void captureThread(bool useCamera, const string& imageFolderPath) {
        // В этом потоке работаем только с CPU данными
        cv::ocl::setUseOpenCL(false); // Отключаем OpenCL для захвата
        
        VideoCapture cap;
        double videoFPS = 15.0; // Значение по умолчанию        
        
        if (useCamera) {
            cap.open(videoSource);
            frameSize = Size(
                static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
                static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT))
            );
            
            if (frameSize.width <= 0 || frameSize.height <= 0) {
                frameSize = Size(640, 480);
                cap.set(CAP_PROP_FRAME_WIDTH, frameSize.width);
                cap.set(CAP_PROP_FRAME_HEIGHT, frameSize.height);
            }
        } else {
            // Для режима изображений загружаем первое изображение
            Mat firstFrameCPU;
            loadImage(firstFrameCPU, 0, imageFolderPath);
            
            if (firstFrameCPU.empty()) {
                cerr << "Cannot find any images in the sequence" << endl;
                running = false;
                return;
            }
            
            frameSize = Size(firstFrameCPU.cols, firstFrameCPU.rows);
            videoFPS = 15.0; // Стандартный FPS для последовательности
        }
        
        // Получаем FPS видеофайла (если это видеофайл, а не камера)
        if (cap.isOpened() && !useCamera) {
            double fpsFromFile = cap.get(CAP_PROP_FPS);
            if (fpsFromFile > 0) {
                videoFPS = fpsFromFile;
                cout << "Video file FPS: " << videoFPS << endl;
            }
        }
        
        // Для видеофайлов включаем замедление для синхронизации с реальным временем
        bool shouldThrottle = !useCamera; // Для файлов и последовательностей изображений
        double frameDelayMs = 1000.0 / videoFPS; // Задержка между кадрами в миллисекундах
        
        cout << "Frame delay: " << frameDelayMs << " ms (" << videoFPS << " FPS)" << endl;
        
        a = frameSize.width;
        b = frameSize.height;
        
        roi.x = static_cast<int>(a * ((1.0 - framePart) / 2.0));
        roi.y = static_cast<int>(b * ((1.0 - framePart) / 2.0));
        roi.width = static_cast<int>(a * framePart);
        roi.height = static_cast<int>(b * framePart);
        
        roi.x = max(0, min(roi.x, a - roi.width));
        roi.y = max(0, min(roi.y, b - roi.height));
        roi.width = min(roi.width, a - roi.x);
        roi.height = min(roi.height, b - roi.y);
        
        cout << "Resolution: " << a << "x" << b << endl;
        cout << "ROI: x=" << roi.x << " y=" << roi.y << " w=" << roi.width << " h=" << roi.height << endl;
        
        int frameId = 0;
        auto lastFpsTime = chrono::steady_clock::now();
        auto lastFrameTime = chrono::steady_clock::now();
        int frameCount = 0;
        int consecutiveSkips = 0;
        
        while (running) {
            auto startTimeCap = chrono::steady_clock::now();
            
            // ========== УПРАВЛЕНИЕ ЗАДЕРЖКОЙ ДЛЯ ВИДЕОФАЙЛОВ ==========
            if (shouldThrottle && frameId > 0) {
                auto now = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFrameTime).count();
                
                if (elapsed < frameDelayMs) {
                    // Спим оставшееся время, чтобы синхронизироваться с реальным временем
                    int sleepMs = static_cast<int>(frameDelayMs - elapsed);
                    if (sleepMs > 0 && sleepMs < 100) {
                        this_thread::sleep_for(chrono::milliseconds(sleepMs));
                    }
                }
            }
            
            int totalLag = rawFramesQueue.size() + processedFramesQueue.size();
            
            // Более мягкая политика пропуска кадров для видеофайлов
            int maxAllowedLag = shouldThrottle ? MAX_PROCESSING_LAG * 2 : MAX_PROCESSING_LAG;
            
            // Если лаг слишком большой, пропускаем кадры
            if (totalLag > maxAllowedLag) {
                consecutiveSkips++;
                
                // Пропускаем кадры при сильной перегрузке
                if (consecutiveSkips > SKIP_FRAMES_THRESHOLD) {
                    framesSkipped++;
                    
                    if (useCamera) {
                        // Для камеры просто читаем и отбрасываем кадр
                        Mat dummy;
                        cap.read(dummy);
                    }
                    // Для видеофайлов - просто пропускаем этот кадр
                    
                    consecutiveSkips = 0;
                }
                
                // Небольшая задержка для снижения нагрузки
                this_thread::sleep_for(chrono::milliseconds(5));
                continue;
            }
            
            consecutiveSkips = 0;
            
            FrameData frameData;
            frameData.frameId = frameId++;
            
            if (useCamera) {
                // Режим камеры - читаем в CPU Mat
                bool frameRead = cap.read(frameData.frameCPU);
                if (!frameRead || frameData.frameCPU.empty()) {
                    //cerr << "Failed to read frame from camera, reopening..." << endl;
                    cap.release();
                    cap.open(videoSource);
                    this_thread::sleep_for(chrono::milliseconds(10));
                    continue;
                }
            } else {
                // Режим видеофайла или последовательности изображений
                if (cap.isOpened()) {
                    // Читаем из видеофайла
                    bool frameRead = cap.read(frameData.frameCPU);
                    if (!frameRead || frameData.frameCPU.empty()) {
                        // Видео закончилось - перематываем на начало (loop)
                        cout << "Video ended, restarting..." << endl;
                        cap.set(CAP_PROP_POS_FRAMES, 0);
                        frameRead = cap.read(frameData.frameCPU);
                        if (!frameRead || frameData.frameCPU.empty()) {
                            cout << "Cannot restart video, stopping..." << endl;
                            running = false;
                            break;
                        }
                    }
                } else {
                    // Режим изображений
                    loadImage(frameData.frameCPU, frameData.frameId % 1200, imageFolderPath);
                    
                    if (frameData.frameCPU.empty()) {
                        cerr << "Failed to load image for frame " << frameData.frameId << endl;
                        loadImage(frameData.frameCPU, frameData.frameId % 1200 + 1, imageFolderPath);
                        if (frameData.frameCPU.empty()) {
                            cout << "Image sequence ended" << endl;
                            running = false;
                            break;
                        }
                        frameData.frameId++;
                    }
                }
            }
            
            if (frameData.frameCPU.empty()) continue;
            
            // Создаем уменьшенную серую версию для обработки (на CPU)
            Mat compressed, gray;
            Size compressedSize(a / compressionConfig, b / compressionConfig);
            if (compressedSize.width <= 0) compressedSize.width = 1;
            if (compressedSize.height <= 0) compressedSize.height = 1;
            
            resize(frameData.frameCPU, compressed, compressedSize, 0, 0, INTER_AREA);
            cvtColor(compressed, gray, COLOR_BGR2GRAY);
            gray.copyTo(frameData.grayCPU); // Сохраняем на CPU
            
            // Для видеофайлов используем неблокирующую push с повторными попытками
            bool pushed = false;
            int retryCount = 0;
            const int MAX_RETRIES = 3;
            
            while (!pushed && retryCount < MAX_RETRIES && running) {
                if (rawFramesQueue.push(move(frameData))) {
                    pushed = true;
                } else {
                    // Очередь заполнена - ждем немного и пробуем снова
                    retryCount++;
                    this_thread::sleep_for(chrono::milliseconds(5));
                    // Восстанавливаем frameData (оно было перемещено при неудачной попытке)
                    if (retryCount < MAX_RETRIES && !pushed) {
                        // Создаем новый FrameData для повторной попытки
                        FrameData newData;
                        newData.frameId = frameData.frameId;
                        frameData.frameCPU.copyTo(newData.frameCPU);
                        frameData.grayCPU.copyTo(newData.grayCPU);
                        newData.timestamp = frameData.timestamp;
                        frameData = move(newData);
                    }
                }
            }
            
            if (!pushed) {
                // Если после всех попыток не удалось добавить - пропускаем кадр
                framesSkipped++;
                if (framesSkipped % 30 == 0) { // Логируем каждые 30 пропущенных кадров
                    cout << "Warning: Dropped frame " << frameData.frameId 
                        << " (queue full). Total skipped: " << framesSkipped << endl;
                }
            }
            
            frameCount++;
            auto now = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFpsTime);
            if (elapsed.count() >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
            }
            
            auto endTimeCap = chrono::steady_clock::now();
            auto durationCap = chrono::duration_cast<chrono::microseconds>(endTimeCap - startTimeCap);
            processingTimeCapture = (processingTimeCapture * 19.0 + durationCap.count() / 1000.0) / 20.0;
            
            // Обновляем время последнего кадра для управления задержкой
            lastFrameTime = chrono::steady_clock::now();
        }
        
        if (cap.isOpened()) cap.release();
        cout << "Capture thread stopped. Total frames captured: " << frameCount 
            << ", Skipped: " << framesSkipped << endl;
    }
    
    // ========================= ПОТОК ДЕТЕКТИРОВАНИЯ И ОТСЛЕЖИВАНИЯ (ТОЛЬКО CPU) =========================
void detectionAndTrackingThread() {
    cout << "Detection and Tracking thread started (CPU only)" << endl;
    
    // Отключаем OpenCL для этого потока - работаем только на CPU
    cv::ocl::setUseOpenCL(false);
    
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    Size winSize(winSizeConfig, winSizeConfig);
    int consecutiveFailures = 0;
    const int MAX_CONSECUTIVE_FAILURES = 10;
    int framesProcessedInThread = 0;
    
    // Счетчик для периодического обновления точек
    int framesSinceLastRedetection = 0;
    const int REDETECTION_INTERVAL = 15; // Рахрешаем обновление точек через REDETECTION_INTERVAL кадров
    
    while (running) {
        FrameData frameData;
        // if (!rawFramesQueue.wait_and_pop(frameData)) {
        //     if (!running) break;
        //     continue;
        // }

        for (int attempt = 0; attempt < 500 && running; attempt++) {
            if (rawFramesQueue.try_pop(frameData)) {
                break;
            }
            this_thread::sleep_for(chrono::milliseconds(10));
        }
        
        if (!running) break;
        
        if (frameData.frameCPU.empty()) {
            continue;
        }

        
        auto startTimeDetTrack = chrono::steady_clock::now();
        framesProcessedInThread++;
        
        // --- ЭТАП 1: ОТСЛЕЖИВАНИЕ ТОЧЕК (всегда делаем сначала) ---
        if (!firstFrameForTracking && !prevPoints.empty() && !frameData.grayCPU.empty() && !prevGrayCPU.empty()) {
            vector<Point2f> nextPoints;
            vector<uchar> status;
            vector<float> err;
            
            try {
                calcOpticalFlowPyrLK(
                    prevGrayCPU, frameData.grayCPU,
                    prevPoints, nextPoints,
                    status, err,
                    winSize, maxLevelConfig,
                    termcrit, 0, 0.001
                );
            } catch (const exception& e) {
                cerr << "Error in optical flow: " << e.what() << endl;
                frameData.grayCPU.copyTo(prevGrayCPU);
                frameData.points.clear();
                prevPoints.clear();
                trackedPoints.store(0);
                
                while (processedFramesQueue.size() >= MAX_QUEUE_SIZE && running) {
                    this_thread::sleep_for(chrono::milliseconds(1));
                }
                processedFramesQueue.push(move(frameData));
                continue;
            }
            
            vector<Point2f> goodNew;
            vector<Point2f> goodOld;
            
            int goodCount = 0;
            for (size_t i = 0; i < status.size(); i++) {
                if (status[i] && err[i] < 50.0) {
                    goodNew.push_back(nextPoints[i]);
                    goodOld.push_back(prevPoints[i]);
                    goodCount++;
                }
            }
            
            if (goodCount >= 6) {
                Mat T;
                try {
                    T = estimateAffine2D(goodOld, goodNew, noArray(), RANSAC, 3.0);
                } catch (const exception& e) {
                    cerr << "Error in estimateAffine2D: " << e.what() << endl;
                    T = Mat();
                }
                
                if (!T.empty() && T.rows == 2 && T.cols == 3) {
                    double dx = T.at<double>(0, 2) * compressionConfig;
                    double dy = T.at<double>(1, 2) * compressionConfig;
                    double da = atan2(T.at<double>(1, 0), T.at<double>(0, 0));
                    
                    frameData.transformFirstDerivative = TransformParam(dx, dy, da);
                } else {
                    frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                }
            } else {
                frameData.transformFirstDerivative = TransformParam(0, 0, 0);
            }
            
            // Сохраняем отслеженные точки
            frameData.points = goodNew;
            trackedPoints.store(static_cast<int>(goodNew.size()));
            consecutiveFailures = 0;
        } else {
            if (consecutiveFailures++ > MAX_CONSECUTIVE_FAILURES) {
                firstFrameForTracking = true;
                consecutiveFailures = 0;
                trackedPoints.store(0);
            }
            frameData.transformFirstDerivative = TransformParam(0, 0, 0);
        }
        
        // --- ЭТАП 2: ДОБАВЛЕНИЕ НОВЫХ ТОЧЕК (если нужно) ---
        int currentPointCount = frameData.points.size();
        bool needMorePoints = currentPointCount < maxCornersConfig * 0.7; // Нужно больше точек если меньше 70% от максимума
        bool timeToRedetect = framesSinceLastRedetection >= REDETECTION_INTERVAL && !firstFrameForTracking;
        
        if (firstFrameForTracking || prevPoints.empty() || needMorePoints && timeToRedetect) {
            
            // Сбрасываем счетчик при детектировании
            framesSinceLastRedetection = 0;
            
            // Детектирование новых точек на CPU
            Mat maskCPU = Mat::zeros(frameData.grayCPU.size(), CV_8U);
            int marginX = frameData.grayCPU.cols / 8;
            int marginY = frameData.grayCPU.rows / 8;
            rectangle(maskCPU, 
                    Rect(marginX, marginY, 
                        frameData.grayCPU.cols - 2 * marginX, 
                        frameData.grayCPU.rows - 2 * marginY),
                    Scalar(255), FILLED);
            
            vector<KeyPoint> keypoints;
            try {
                detector->detect(frameData.grayCPU, keypoints, maskCPU);
            } catch (const exception& e) {
                cerr << "Error in detector: " << e.what() << endl;
                continue;
            }
            
            // Создаем маску для исключения областей с существующими точками
            Mat exclusionMask = Mat::zeros(frameData.grayCPU.size(), CV_8U);
            if (!frameData.points.empty()) {
                for (const auto& pt : frameData.points) {
                    circle(exclusionMask, pt, static_cast<int>(minDistanceConfig), Scalar(255), FILLED);
                }
            }
            
            // Вычисляем сколько новых точек нужно добавить (максимум 15% от максимума)
            int maxNewPoints = max(1, static_cast<int>(maxCornersConfig * 0.15));
            int targetNewPoints = min(maxNewPoints, maxCornersConfig - static_cast<int>(frameData.points.size()));
            
            if (targetNewPoints > 0 && !keypoints.empty()) {
                // Сортируем ключевые точки по качеству (силе отклика)
                sort(keypoints.begin(), keypoints.end(), 
                     [](const KeyPoint& a, const KeyPoint& b) {
                         return a.response > b.response;
                     });
                
                // Выбираем новые точки, которые не попадают в зоны существующих
                vector<Point2f> newPoints;
                for (const auto& kp : keypoints) {
                    if (newPoints.size() >= targetNewPoints) break;
                    
                    int x = static_cast<int>(kp.pt.x);
                    int y = static_cast<int>(kp.pt.y);
                    
                    // Проверяем границы
                    if (x >= 0 && x < exclusionMask.cols && y >= 0 && y < exclusionMask.rows) {
                        // Проверяем, не попадает ли точка в зону исключения
                        if (exclusionMask.at<uchar>(y, x) == 0) {
                            newPoints.push_back(kp.pt);
                            // Добавляем в маску исключения, чтобы не брать близкие точки
                            circle(exclusionMask, kp.pt, static_cast<int>(minDistanceConfig), Scalar(255), FILLED);
                        }
                    }
                }
                
                // Добавляем новые точки к существующим
                if (!newPoints.empty()) {
                    frameData.points.insert(frameData.points.end(), newPoints.begin(), newPoints.end());
                    // Удаляем слишком близкие точки и ограничиваем до максимума
                    removeFramePoints(frameData.points, minDistanceConfig * 0.8);
                    if (frameData.points.size() > maxCornersConfig) {
                        frameData.points.resize(maxCornersConfig);
                    }
                }
            }
            
            if (firstFrameForTracking) {
                // Первый кадр - просто сохраняем точки
                prevPoints = frameData.points;
                frameData.grayCPU.copyTo(prevGrayCPU);
                firstFrameForTracking = false;
                trackedPoints.store(static_cast<int>(prevPoints.size()));
            }
        } else {
            // Если не детектируем, увеличиваем счетчик
            framesSinceLastRedetection++;
        }
        
        // Обновляем prevPoints для следующего кадра (используем обновленный frameData.points)
        if (!firstFrameForTracking) {
            prevPoints = frameData.points;
            frameData.grayCPU.copyTo(prevGrayCPU);
        }
        
        // Адаптивная настройка параметров детектора
        if ((frameData.points.size() < maxCornersConfig / 6)) {
            qualityLevelConfig *= 0.98;
            harrisKConfig *= 0.98;
            detector->setQualityLevel(qualityLevelConfig);
            detector->setK(harrisKConfig);
        }
        if ((frameData.points.size() > maxCornersConfig * 4 / 5)) {
            qualityLevelConfig *= 1.02;
            harrisKConfig *= 1.02;
            detector->setQualityLevel(qualityLevelConfig);
            detector->setK(harrisKConfig);
        }

        auto endTimeDetTrack = chrono::steady_clock::now();
        auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
        processingTimeDetectionTracking = (processingTimeDetectionTracking * 99.0 + durationDetTrack.count() / 1000.0) / 100.0;
        
        // Ждем, если очередь переполнена
        while (processedFramesQueue.size() >= MAX_QUEUE_SIZE && running) {
            this_thread::sleep_for(chrono::milliseconds(5));
        }
        
        processedFramesQueue.push(move(frameData));
    }
    
    cout << "Detection and Tracking thread stopped. Processed " << framesProcessedInThread << " frames." << endl;
}
    
    // ========================= ПОТОК СТАБИЛИЗАЦИИ (ТОЛЬКО ЗДЕСЬ ИСПОЛЬЗУЕТСЯ GPU) =========================
    void stabilizationThread() {
        cout << "Stabilization thread started (GPU enabled)" << endl;
        
        // Включаем OpenCL только для этого потока
        cv::ocl::setUseOpenCL(USE_OPENCL);
        
        int framesStabilized = 0;
        
        while (running) {
            FrameData frameData;
            if (!processedFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }

            auto startTimeStab = chrono::steady_clock::now();
            
            if (!frameData.frameCPU.empty()) {
                try {
                    if (kSwitch < 0.01) kSwitch = 0.01;
                    if (kSwitch < 1.0) {
                        kSwitch *= 1.06;
                        kSwitch += 0.005;
                    } else if (kSwitch > 1.0) {
                        kSwitch = 1.0;
                    }
                    
                    kSwitch = 1.0;
                    framesProcessed++;
                    framesStabilized++;
                    
                    // Стабилизация (вычисление матрицы трансформации на CPU)
                    iirAdaptive(frameData.transformFirstDerivative, oldTransform, 
                            frameData.transformSKO, frameData.stabMatrix, tauStab, roi, a, b, kSwitch);
                    
                    frameData.transform = oldTransform;
                    
                    // Рисуем точки на CPU после получения с GPU
                    if (!frameData.points.empty() && framePart > 0.7) {
                        int pointsToShow = min(300, static_cast<int>(frameData.points.size()));
                        for (int i = 0; i < pointsToShow; i++) {
                            Point2f pt = frameData.points[i];
                            Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                        static_cast<int>(pt.y * compressionConfig));
                            circle(frameData.frameCPU, scaledPt, 6, colorRED, -1);
                        }
                    }
                    
                    // ==== ТОЛЬКО ЗДЕСЬ ИСПОЛЬЗУЕМ GPU ====
                    // Копируем кадр на GPU
                    frameData.frameCPU.copyTo(frameData.frameGPU);
                    double LEN, THETA;
                        LEN = sqrt(frameData.transformFirstDerivative.dx * frameData.transformFirstDerivative.dx + 
                            frameData.transformFirstDerivative.dy * frameData.transformFirstDerivative.dy)*D;

                        if (frameData.transformFirstDerivative.dx == 0.0)
                            if (frameData.transformFirstDerivative.dy > 0.0)
                                THETA = 90.0;
                            else
                                THETA = -90.0;
                        else
                            THETA = atan(frameData.transformFirstDerivative.dy / frameData.transformFirstDerivative.dx) * RAD_TO_DEG;
                    
                    // ==== здесь выполняем винеровскую фильтрацию (только если включена) ====
                    if (framePart < 0.65 && wiener)
                    {
                        UMat zeroMatH(cv::Size(a, b), CV_32F, Scalar(0)), complexH;
                        vector<UMat> gChannels(3), gChannelsWiener(3);

                        
                        UMat uFrame32F;
                        frameData.frameGPU.convertTo(uFrame32F, CV_32F);
                        if (uFrame32F.empty()) {
                            cerr << "[Wiener ERROR] uFrame32F is empty!" << endl;
                            continue;
                        }
                        
                        // PSF фильтр
                        //Size psfSize = cv::Size((int)LEN * 1 + 10, (int)LEN * 1 + 10);
                        Size psfSize = cv::Size(31, 31);
                        GcalcPSF(gH, uFrame32F.size(), psfSize, LEN, THETA);
                        
                        if (gH.empty()) {
                            cerr << "[Wiener ERROR] gH (PSF) is empty!" << endl;
                            continue;
                        }
                        
                        // Отображаем PSF (один раз для первого кадра)
                        static bool psfDisplayed = true;
                        if (!psfDisplayed) {
                            Mat hCpuDisplay;
                            gH.copyTo(hCpuDisplay);
                            Mat hDisplay;
                            normalize(hCpuDisplay, hDisplay, 0, 255, NORM_MINMAX);
                            hDisplay.convertTo(hDisplay, CV_8U);
                            imshow("Wiener_PSF_Filter", hDisplay);
                            psfDisplayed = true;
                        }
                        
                        // Вычисляем Wiener фильтр
                        GcalcWnrFilter(gH, gHw, nsr);
                        
                        if (gHw.empty()) {
                            cerr << "[Wiener ERROR] gHw is empty!" << endl;
                            continue;
                        }
                        
                        // Отображаем частотную характеристику (один раз)
                        static bool wnrDisplayed = true;
                        if (!wnrDisplayed) {
                            Mat wnrCpuDisplay;
                            gHw.copyTo(wnrCpuDisplay);
                            Mat wnrDisplay;
                            normalize(wnrCpuDisplay, wnrDisplay, 0, 255, NORM_MINMAX);
                            wnrDisplay.convertTo(wnrDisplay, CV_8U);
                            imshow("Wiener_Freq_Response", wnrDisplay);
                            wnrDisplayed = true;
                        }
                        
                        // Объединяем действительную и мнимую часть фильтра
                        vector<cv::UMat> planesH = { gHw, zeroMatH };
                        cv::merge(planesH, complexH);
                        
                        if (complexH.empty()) {
                            cerr << "[Wiener ERROR] complexH is empty!" << endl;
                            continue;
                        }
                        
                        // Разделяем каналы
                        split(uFrame32F, gChannels);
                        
                        // Обработка трех цветных каналов
                        for (unsigned short i = 0; i < 3; i++) {
                            if (!gChannels[i].empty()) {
                                Gfilter2DFreq(gChannels[i], gChannelsWiener[i], complexH);
                            }
                        }
                        
                        // Объединяем обратно
                        cv::merge(gChannelsWiener, uFrame32F);
                        
                        if (uFrame32F.empty()) {
                            cerr << "[Wiener ERROR] uFrame32F is empty after merge!" << endl;
                            continue;
                        }
                        
                        // Сохраняем результат обратно в frameGPU
                        uFrame32F.convertTo(frameData.frameGPU, CV_8UC3);
                        
                        if (frameData.frameGPU.empty()) {
                            cerr << "[Wiener ERROR] frameGPU is empty after Wiener!" << endl;
                        }
                    }
                    
                    // ==== СТАБИЛИЗАЦИЯ ====
                    UMat stabilizedFrame, croppedFrame;
                    warpAffineOptimized(frameData.frameGPU, stabilizedFrame, frameData.stabMatrix, frameSize, 
                                       cv::INTER_LINEAR, cv::BORDER_CONSTANT, Scalar(), useFP16Warp);
                    croppedFrame = stabilizedFrame(roi);
                    
                    // Копируем результат обратно на CPU для отображения
                    croppedFrame.copyTo(frameData.frameCPU);
                    // ====================================
                    
                } catch (const exception& e) {
                    cerr << "Error in stabilization: " << e.what() << endl;
                }
            }
            
            {
                lock_guard<mutex> lock(bufferMutex);
                
                // Ограничиваем размер буфера
                if (frameBuffer.size() >= MAX_FRAME_BUFFER_SIZE) {
                    int oldestFrameId = frameBuffer.begin()->first;
                    frameBuffer.erase(oldestFrameId);
                }
                
                frameBuffer[frameData.frameId] = move(frameData);
                frameReadyCV.notify_one();
                
                // Дополнительная очистка старых кадров
                if (frameBuffer.size() > MAX_FRAME_BUFFER_SIZE) {
                    vector<int> toRemove;
                    for (auto& pair : frameBuffer) {
                        if (pair.first < nextDisplayFrameId - MAX_FRAME_BUFFER_SIZE) {
                            toRemove.push_back(pair.first);
                        }
                    }
                    for (int id : toRemove) {
                        frameBuffer.erase(id);
                    }
                }
            }

            auto endTimeStab = chrono::steady_clock::now();
            auto durationStab = chrono::duration_cast<chrono::microseconds>(endTimeStab - startTimeStab);
            processingTimeStabilization = (processingTimeStabilization * 19.0 + durationStab.count() / 1000.0) / 20.0;
        }
        
        cout << "Stabilization thread stopped. Stabilized " << framesStabilized << " frames total." << endl;
    }
    
    // ========================= ПОТОК ОТОБРАЖЕНИЯ =========================
    void displayThread() {
        cout << "Display thread started" << endl;
        
        // Отключаем OpenCL для отображения
        cv::ocl::setUseOpenCL(false);
        
        const string windowName = "Video Stabilization";
        namedWindow(windowName, WINDOW_NORMAL);
        resizeWindow(windowName, 1280, 720);
        
        VideoWriter writer;
        if (recordEnable) {
            writer.open("stabilized_output.mp4", 
                       VideoWriter::fourcc('a', 'v', 'c', '1'),
                       30, frameSize);
        }
        
        int displayedFrames = 0;
        auto lastDisplayTime = chrono::steady_clock::now();
        auto lastFrameTime = chrono::steady_clock::now();
        
        while (running) {
            auto startTimeDisp = chrono::steady_clock::now();
            FrameData frameData;
            bool gotFrame = false;
            
            {
                unique_lock<mutex> lock(bufferMutex);
                
                auto it = frameBuffer.find(nextDisplayFrameId);
                if (it != frameBuffer.end()) {
                    frameData = move(it->second);
                    frameBuffer.erase(it);
                    nextDisplayFrameId++;
                    gotFrame = true;
                } else {
                    // Если нет нужного кадра, пробуем найти ближайший
                    if (!frameBuffer.empty()) {
                        int nextAvailable = -1;
                        for (auto& pair : frameBuffer) {
                            if (pair.first >= nextDisplayFrameId) {
                                if (nextAvailable == -1 || pair.first < nextAvailable) {
                                    nextAvailable = pair.first;
                                }
                            }
                        }
                        
                        if (nextAvailable != -1 && nextAvailable - nextDisplayFrameId < 3) {
                            frameData = move(frameBuffer[nextAvailable]);
                            frameBuffer.erase(nextAvailable);
                            nextDisplayFrameId = nextAvailable + 1;
                            gotFrame = true;
                        } else if (!frameBuffer.empty()) {
                            // Если разрыв слишком большой, пропускаем кадры
                            nextDisplayFrameId = frameBuffer.begin()->first;
                            continue;
                        }
                    }
                    
                    if (!gotFrame) {
                        frameReadyCV.wait_for(lock, chrono::milliseconds(5));
                        continue;
                    }
                }
            }
            
            if (!gotFrame || frameData.frameCPU.empty()) {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            
            displayedFrames++;
            
            // Отображаем прямо с CPU Mat
            Mat displayFrame;
            resize(frameData.frameCPU, displayFrame, Size(a, b), INTER_AREA);
            
            string infoText = format("FPS: %d | nsr: %1.2f | D: %1.2f | part: %1.2f | Skip: %d",
                                    fps.load(),
                                    nsr, 
                                    D, framePart, framesSkipped.load());

            string infoLatencies = format("Cap: %2.1f | D+T: %2.1f | Stab: %2.1f | Q: %ld / %ld ",
                                    processingTimeCapture, processingTimeDetectionTracking, 
                                    processingTimeStabilization,
                                    rawFramesQueue.size(), processedFramesQueue.size());

            putText(displayFrame, infoText, Point(10, 50 * a / 800),
                   FONT_HERSHEY_SIMPLEX, 0.5 * a / 800, colorBLUE, 2 * a / 800);
            putText(displayFrame, infoLatencies, Point(10, 100 * a / 800),
                   FONT_HERSHEY_SIMPLEX, 0.5 * a / 800, colorBLUE, 2 * a / 800);
            
            imshow(windowName, displayFrame);
            
            auto endTimeDisp = chrono::steady_clock::now();
            auto durationDisp = chrono::duration_cast<chrono::microseconds>(endTimeDisp - startTimeDisp);
            processingTimeImshow = (processingTimeImshow * 19.0 + durationDisp.count() / 1000.0) / 20.0;

            if (recordEnable && writer.isOpened()) {
                writer.write(displayFrame);
            }
            
            auto now = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastDisplayTime);
            if (elapsed.count() >= 1000) {
                displayedFrames = 0;
                lastDisplayTime = now;
            }
            
            auto frameElapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFrameTime);
            if (frameElapsed.count() < 10) { // Ограничение до ~100 FPS
                this_thread::sleep_for(chrono::milliseconds(10 - frameElapsed.count()));
            }
            lastFrameTime = now;
            
            int key = waitKey(1);
            if (key == 27 || key == 'q') {
                cout << "Exit requested by user" << endl;
                running = false;
                break;
            } else if (key == ' ') {
                cout << "Paused. Press any key to continue..." << endl;
                waitKey(0);
            } else if (key == 'f') {
                string filename = format("frame_%06d.jpg", frameData.frameId);
                imwrite(filename, displayFrame);
                cout << "Frame saved: " << filename << endl;
            } else if (key == 'd') {
                debugMode = !debugMode;
                cout << "Debug mode: " << (debugMode ? "ON" : "OFF") << endl;
            } else if (key == 's' || key == 'S') {
                if (framePart < 0.95) {
                    framePart *= 1.01;
                    if (framePart > 0.9) framePart = 0.9;
                    roi.x = a * ((1.0 - framePart) / 2.0);
                    roi.y = b * ((1.0 - framePart) / 2.0);
                    roi.width = a * framePart;
                    roi.height = b * framePart;
                }
            } else if (key == 'w' || key == 'W') {
                if (framePart > 0.2) {
                    framePart *= 0.99;
                    if (framePart < 0.05) framePart = 0.05;
                    roi.x = a * ((1.0 - framePart) / 2.0);
                    roi.y = b * ((1.0 - framePart) / 2.0);
                    roi.width = a * framePart;
                    roi.height = b * framePart;
                }
            } else if (key == 'u' || key == 'U') {
                D = D + 0.01;
                if (D > 1.0)
                    D = 1.0;
            } else if (key == 'i' || key == 'I') {
                D = D - 0.01;
                if (D < 0.0)
                    D = 0.01;
            } else if (key == 'j' || key == 'J') {
                nsr = nsr*2;
                if (nsr > 10.0)
                    nsr = 10.0;
            } else if (key == 'k' || key == 'K') {
                nsr = nsr*0.5;
                if (nsr < 0.01)
                    nsr = 0.01;
            }
        }
        
        if (writer.isOpened()) writer.release();
        destroyWindow(windowName);
        cout << "Display thread stopped" << endl;
    }
    
    // ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================
    void loadImage(Mat& image_cpu, int frame_id, const string& filepath) {
        char file[200];
        
        sprintf(file, "image_0/%06d.png", frame_id);
        string filename = filepath + string(file);
        
        ifstream file_check(filename.c_str());
        if (file_check.good()) {
            file_check.close();
            image_cpu = imread(filename, IMREAD_COLOR);
            if (!image_cpu.empty()) return;
        }
        
        sprintf(file, "image_0/%06d.jpg", frame_id);
        filename = filepath + string(file);
        
        ifstream jpg_check(filename.c_str());
        if (jpg_check.good()) {
            jpg_check.close();
            image_cpu = imread(filename, IMREAD_COLOR);
        }
    }

    void removeFramePoints(vector<Point2f>& p0, double minDistance) {
        if (p0.empty()) return;

        sort(p0.begin(), p0.end(), [](const Point2f& a, const Point2f& b) {
            return a.x < b.x;
        });

        vector<bool> toRemove(p0.size(), false);
        for (size_t i = 0; i < p0.size(); ++i) {
            if (toRemove[i]) continue; 

            for (size_t j = i + 1; j < p0.size(); ++j) {
                if (p0[j].x - p0[i].x > minDistance) {
                    break; 
                }

                float dx = p0[j].x - p0[i].x;
                float dy = p0[j].y - p0[i].y;
                float distanceSq = dx * dx + dy * dy;

                if (distanceSq < minDistance * minDistance) {
                    toRemove[j] = true;
                }
            }
        }

        for (int i = p0.size() - 1; i >= 0; --i) {
            if (toRemove[i]) {
                p0.erase(p0.begin() + i);
            }
        }
    }

    void iirAdaptive(TransformParam& transformsFirtsDerivative, TransformParam& transforms, TransformParam& transformSKO, UMat& stabMatrix, 
        double& tauStab, Rect& roi, const int a, const int b, double& kSwitch) {
        if (abs(transformsFirtsDerivative.dx) < 4.0 * transformSKO.dx + a/8) {
            transforms.dx = kSwitch * (transforms.dx * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.dx);
        } 

        if (abs(transformsFirtsDerivative.dy) < 4.0 * transformSKO.dy + b/8) {
            transforms.dy = kSwitch * (transforms.dy * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.dy);
        }

        if (abs(transformsFirtsDerivative.da) < 4.0 * transformSKO.da + 0.1) {
            transforms.da = kSwitch * (transforms.da * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.da);
        }

        if (transforms.da > CV_PI) transforms.da -= CV_PI;
        if (transforms.da < -CV_PI) transforms.da += CV_PI;

        if (tauStab < TAU_STAB_MAX/4) tauStab *= 1.2;
        if (tauStab < TAU_STAB_MAX/2 && !(abs(transforms.dx) > a / 2 || abs(transforms.dy) > b / 2)) tauStab *= 1.1;
        if (tauStab < TAU_STAB_MAX && !(abs(transforms.dx) > a / 3 || abs(transforms.dy) > b / 3)) {
            tauStab *= 1.1;
            if (tauStab > TAU_STAB_MAX) tauStab = TAU_STAB_MAX;
        }

        if (roi.x + (int)transforms.dx < 0) {
            transforms.dx = double(1 - roi.x);
            if (tauStab > TAU_STAB_MAX/2) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        } else if (roi.x + roi.width + (int)transforms.dx >= a) {
            transforms.dx = (double)(a - roi.x - roi.width);
            if (tauStab > TAU_STAB_MAX/2) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (roi.y + (int)transforms.dy < 0) {
            transforms.dy = (double)(1 - roi.y);
            if (tauStab > TAU_STAB_MAX/8) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        } else if (roi.y + roi.height + (int)transforms.dy >= b) {
            transforms.dy = (double)(b - roi.y - roi.height);
            if (tauStab > TAU_STAB_MAX/2) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (kSwitch < 1.0) tauStab *= (4.0 + kSwitch) / 5.0;

        transformSKO.dx = (1.0 - 0.1) * transformSKO.dx + 0.1 * abs(transformsFirtsDerivative.dx);
        transformSKO.dy = (1.0 - 0.1) * transformSKO.dy + 0.1 * abs(transformsFirtsDerivative.dy);
        transformSKO.da = (1.0 - 0.1) * transformSKO.da + 0.1 * abs(transformsFirtsDerivative.da);

        transforms.getTransformInvert(stabMatrix);
    }
  

    //WITH OPENCL
    void GcalcPSF(cv::UMat& outputImg, Size filterSize, Size psfSize, double len, double theta)
    {
        int scale = 4;
        cv::UMat h(filterSize, CV_32F, Scalar(0));
        Mat hCpu(psfSize, CV_32F, Scalar(0));
        Mat hCpuBig(Size(psfSize.width * scale, psfSize.height * scale), CV_32F, Scalar(0));
        Point center(psfSize.width * scale / 2, psfSize.height * scale / 2);

        Size axes(scale, cvRound(double(len * scale ) * 0.5f));
        Size axes2(scale, cvRound(double(len * scale) * 0.45f));
        Size axes3(scale, cvRound(double(len * scale) * 0.3f));
        
        double angle = 90.0 - theta;

        ellipse(hCpuBig, center, axes, angle, 0, 360, Scalar(0.6), FILLED);
        ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.9), FILLED);
        ellipse(hCpuBig, center, axes3, angle, 0, 360, Scalar(1.0), FILLED);
        resize(hCpuBig, hCpu, psfSize, INTER_LINEAR);
        if (hCpu.cols > h.cols / 2)
            resize(hCpu, hCpu, Size(h.cols / 2 - 1, hCpu.rows), INTER_LINEAR);
        if (hCpu.rows > h.rows / 2)
            resize(hCpu, hCpu, Size(hCpu.cols, h.rows / 2 - 1), INTER_LINEAR);
        
        //imshow("PSF Cpu", hCpu);
        //hCpu(Rect(0, 0, psfSize.width, psfSize.height)).copyTo(h(Rect((filterSize.width - psfSize.width) / 2, (filterSize.height - psfSize.height) / 2, psfSize.width, psfSize.height)));
        hCpu(Rect(0, 0, hCpu.cols, hCpu.rows)).copyTo(h(Rect((filterSize.width - hCpu.cols) / 2, (filterSize.height - hCpu.rows) / 2, hCpu.cols, hCpu.rows)));

        Scalar summa = cv::sum(h);

        cv::divide(h, Scalar(summa[0]), outputImg);

    }

    void Gfftshift(const cv::UMat& inputImg, cv::UMat& outputImg)
    {
        outputImg = inputImg.clone();
        int cx = outputImg.cols / 2;
        int cy = outputImg.rows / 2;
        cv::UMat q0(outputImg, Rect(0, 0, cx, cy));
        cv::UMat q1(outputImg, Rect(cx, 0, cx, cy));
        cv::UMat q2(outputImg, Rect(0, cy, cx, cy));
        cv::UMat q3(outputImg, Rect(cx, cy, cx, cy));
        cv::UMat tmp;
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);
        q1.copyTo(tmp);
        q2.copyTo(q1);
        tmp.copyTo(q2);
    }

    void Gfilter2DFreq(const cv::UMat& inputImg, cv::UMat& outputImg, const cv::UMat& complexH)
    {
        cv::UMat zeroMat(inputImg.size(), CV_32F, Scalar(0));


        vector<cv::UMat> planes = { inputImg, zeroMat };
        cv::UMat complexInput;
        cv::merge(planes, complexInput);
        dft(complexInput, complexInput, DFT_COMPLEX_OUTPUT | DFT_SCALE);
        cv::UMat complexOutput;
        cv::mulSpectrums(complexInput, complexH, complexOutput, 0);
        dft(complexOutput, complexOutput, DFT_INVERSE | DFT_COMPLEX_INPUT);
        vector<cv::UMat> planesOut;
        cv::split(complexOutput, planesOut);
        outputImg = planesOut[0];
    }

    void GcalcWnrFilter(const cv::UMat& input_h_PSF, cv::UMat& output_G, double nsr)
    {
        cv::UMat h_PSF_shifted;
        Gfftshift(input_h_PSF, h_PSF_shifted);
        cv::UMat zeroMat(h_PSF_shifted.size(), CV_32F, Scalar(0));
        vector<cv::UMat> planes = { h_PSF_shifted, zeroMat };
        cv::UMat complexI;
        cv::merge(planes, complexI);

        cv::dft(complexI, complexI);
        vector<cv::UMat> planesOut;
        cv::split(complexI, planesOut);
        cv::UMat denom;
        cv::magnitude(planesOut[0], planesOut[1], denom);
        cv::pow(denom, 2, denom);
        cv::add(denom, nsr, denom);
        cv::divide(planesOut[0], denom, output_G);
    }

    void Gedgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
    {
        int Nx = inputImg.cols;
        int Ny = inputImg.rows;
        Mat w1(1, Nx, CV_32F, Scalar(0));
        Mat w2(Ny, 1, CV_32F, Scalar(0));

        double* p1 = w1.ptr<double>(0);
        double* p2 = w2.ptr<double>(0);
        double dx = double(2.0 * CV_PI / Nx);
        double x = double(-CV_PI);
        for (int i = 0; i < Nx; i++)
        {
            p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
            x += dx;
        }
        double dy = double(2.0 * CV_PI / Ny);
        double y = double(-CV_PI);
        for (int i = 0; i < Ny; i++)
        {
            p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
            y += dy;
        }
        Mat w = w2 * w1;
        multiply(inputImg, w, outputImg);
    }

    void channelWiener(const cv::UMat* gChannel, cv::UMat* gChannelWiener,
        const cv::UMat* complexH)
    {
        Gfilter2DFreq(*gChannel, *gChannelWiener, *complexH);
    }

};



// ========================= ОСНОВНАЯ ФУНКЦИЯ =========================

int main() {

    cout << cv::getBuildInformation() << std::endl;
    cout << "========================================" << endl;
    cout << " MULTI-THREADED VIDEO STABILIZER OPENCL " << endl;
    cout << "========================================" << endl;
    
    cv::ocl::setUseOpenCL(USE_OPENCL);
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    
    // Обнаружить возможности GPU
    cout << "\n[ИНИЦИАЛИЗАЦИЯ GPU]" << endl;
    GPUCapabilities gpuCaps = detectGPUCapabilities();
    bool useFP16 = gpuCaps.supportsFP16 || gpuCaps.supportsHalfType;
    if (!useFP16) {
        cout << "[GPU] Предупреждение: FP16 недоступен, используется стандартная обработка" << endl;
    }
    cout << endl;
    
    cout << "Выберите режим работы:" << endl;
    cout << "0. Использовать http:// камеру" << endl;
    cout << "1. Использовать камеру" << endl;
    cout << "2. Читать кадры из папки PXL_3" << endl;
    cout << "3. Читать кадры из папки PXL_4K" << endl;
    cout << "4. Читать видео PXL_1 из PC" << endl;
    cout << "Введите 0, 1, 2 или 3: ";
    
    int choice;
    cin >> choice;
    
    bool useCamera = true;
    string imageFolderPath;
    
    if (choice == 2) {
        useCamera = false;
        cout << "Введите путь к папке с кадрами: ";
        cin.ignore();
        getline(cin, imageFolderPath);
        
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        if (imageFolderPath.empty()) {
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_3/";
        }
        cout << "Путь к кадрам: " << imageFolderPath << endl;
    } else if (choice == 3) {
        useCamera = false;
        cout << "Введите путь к папке с кадрами: ";
        cin.ignore();
        getline(cin, imageFolderPath);
        
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        if (imageFolderPath.empty()) {
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_4K/";
        }
        cout << "Путь к кадрам: " << imageFolderPath << endl;
    } else if (choice == 0){
        cout << "Используется режим http камеры по умолчанию:" << "http://192.168.0.103:4747/video?500x500" << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "http://192.168.0.105:4747/video?500x500";
        }
    } else if (choice == 4){
        cout << "Используется режим чтения видео по умолчанию: /home/selbizo/CV/dataset/videos/PXL_1.mp4" << videoSource << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/home/selbizo/CV/dataset/videos/PXL_1.mp4";
        }
    } else {
        cout << "Используется режим камеры по умолчанию: " << "/home/pi/opencv_projects/videos/PXL_1.mp4" << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";
        }
    }
    
    VideoStabilizer stabilizer;
    stabilizer.setUseFP16(useFP16);  // Установить флаг FP16 оптимизации
    stabilizer.start(useCamera, imageFolderPath);
    
    cout << endl << "Управление:" << endl;
    cout << "  ESC или Q - выход" << endl;
    cout << "  Пробел - пауза" << endl;
    cout << "  F - сохранить текущий кадр" << endl;
    cout << "  D - переключить режим отладки" << endl;
    cout << "  S/W - увеличить/уменьшить область кадра" << endl;
    cout << "  U/I - увеличить/уменьшить коэффициент заполнения" << endl;
    cout << "  J/K - увеличить/уменьшить NSR" << endl;
    if (useFP16) {
        cout << "\n[GPU] FP16 оптимизация активирована - быстрее, но возможна небольшая потеря качества" << endl;
    }
    
    try {
        while (stabilizer.running) {
            this_thread::sleep_for(chrono::seconds(1));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}