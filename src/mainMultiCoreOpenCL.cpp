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
using namespace cv;
using namespace std;
namespace fs = filesystem;

// ========================= КОНСТАНТЫ И КОНФИГУРАЦИЯ =========================
const bool USE_OPENCL = true;

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
const int compressionConfig = 3; // Сжатие для обработки
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

// Источник видео
string videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";

// ========================= НОВЫЕ КОНСТАНТЫ ДЛЯ УПРАВЛЕНИЯ ПАМЯТЬЮ =========================
const int MAX_QUEUE_SIZE = 5;           // Максимальный размер очередей
const int MAX_FRAME_BUFFER_SIZE = 7;    // Максимальный размер буфера кадров для отображения
const int SKIP_FRAMES_THRESHOLD = 2;    // Сколько кадров пропускать при переполнении

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
    
    // ========================= ПОТОК ЗАХВАТА КАДРОВ =========================
    void captureThread(bool useCamera, const string& imageFolderPath) {
        // В этом потоке работаем только с CPU данными
        cv::ocl::setUseOpenCL(false); // Отключаем OpenCL для захвата
        
        VideoCapture cap;
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
        }
        
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
        int frameCount = 0;
        int consecutiveSkips = 0;
        
        while (running) {
            auto startTimeCap = chrono::steady_clock::now();
            int totalLag = rawFramesQueue.size() + processedFramesQueue.size();
            
            // Если лаг слишком большой, пропускаем кадры
            if (totalLag > MAX_PROCESSING_LAG) {
                consecutiveSkips++;
                
                // Пропускаем несколько кадров подряд при сильной перегрузке
                if (consecutiveSkips > SKIP_FRAMES_THRESHOLD) {
                    //cout << "High load, skipping frames..." << endl;
                    framesSkipped += SKIP_FRAMES_THRESHOLD;
                    
                    if (useCamera) {
                        // Для камеры просто читаем и отбрасываем кадр
                        Mat dummy;
                        cap.read(dummy);
                    }
                    consecutiveSkips = 0;
                }
                
                this_thread::sleep_for(chrono::milliseconds(2));
                continue;
            }
            
            consecutiveSkips = 0;
            
            FrameData frameData;
            frameData.frameId = frameId++;
            
            if (useCamera) {
                // Режим камеры - читаем в CPU Mat
                bool frameRead = cap.read(frameData.frameCPU);
                if (!frameRead || frameData.frameCPU.empty()) {
                    cerr << "Failed to read frame from camera, reopening..." << endl;
                    cap.release();
                    cap.open(videoSource);
                    this_thread::sleep_for(chrono::milliseconds(10));
                    continue;
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
            
            if (frameData.frameCPU.empty()) continue;
            
            // Создаем уменьшенную серую версию для обработки (на CPU)
            Mat compressed, gray;
            Size compressedSize(a / compressionConfig, b / compressionConfig);
            if (compressedSize.width <= 0) compressedSize.width = 1;
            if (compressedSize.height <= 0) compressedSize.height = 1;
            
            resize(frameData.frameCPU, compressed, compressedSize, 0, 0, INTER_AREA);
            cvtColor(compressed, gray, COLOR_BGR2GRAY);
            gray.copyTo(frameData.grayCPU); // Сохраняем на CPU
            
            // Пытаемся отправить в очередь для детекции/трекинга
            if (!rawFramesQueue.push(move(frameData))) {
                // Если очередь переполнена, пропускаем кадр
                framesSkipped++;
                // Освобождаем память кадра, который не удалось добавить
                // (frameData будет автоматически уничтожен при выходе из области видимости)
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
        }
        
        if (useCamera) cap.release();
        cout << "Capture thread stopped" << endl;
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
        if (!rawFramesQueue.wait_and_pop(frameData)) {
            if (!running) break;
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
            
            // Вычисляем сколько новых точек нужно добавить (максимум 5% от максимума)
            int maxNewPoints = max(1, static_cast<int>(maxCornersConfig * 0.05));
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
                    
                    // Рисуем точки на CPU перед отправкой на GPU
                    if (!frameData.points.empty()) {
                        int pointsToShow = min(300, static_cast<int>(frameData.points.size()));
                        for (int i = 0; i < pointsToShow; i++) {
                            Point2f pt = frameData.points[i];
                            Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                        static_cast<int>(pt.y * compressionConfig));
                            circle(frameData.frameCPU, scaledPt, 2, colorRED, -1);
                        }
                    }
                    
                    // ==== здесь выполняем винеровскую фильтрацию
                    


                    // ==== ТОЛЬКО ЗДЕСЬ ИСПОЛЬЗУЕМ GPU ====
                    // Копируем кадр на GPU только для warpAffine
                    frameData.frameCPU.copyTo(frameData.frameGPU);
                    
                    UMat stabilizedFrame, croppedFrame;
                    warpAffine(frameData.frameGPU, stabilizedFrame, frameData.stabMatrix, frameSize, cv::INTER_CUBIC, cv::BORDER_REPLICATE);
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
                    // Находим самый старый кадр и заменяем его
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
            
            // Небольшая задержка для снижения нагрузки на GPU
            //this_thread::sleep_for(chrono::milliseconds(1));
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
            
            string infoText = format("FPS: %d | points: %d | tau: %2.1f | part: %1.2f | Skip: %d",
                                    fps.load(),
                                    frameData.points.size(), 
                                    tauStab, framePart, framesSkipped.load());

            string infoLatencies = format("Cap: %2.1f | D+T: %2.1f | Stab: %2.1f | Q: %d / %d ",
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
                this_thread::sleep_for(chrono::milliseconds(33 - frameElapsed.count()));
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
};

void calcPSF(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(0, cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];

	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);
}

void calcPSF_circle(Mat& outputImg, Size filterSize, int len, double theta)
{
	Mat h(filterSize, CV_32F, Scalar(0));
	Point point(filterSize.width / 2, filterSize.height / 2);
	ellipse(h, point, Size(cvRound(double(len) / 2.0), cvRound(double(len) / 2.0)), 90.0 - theta, 0, 360, Scalar(255), FILLED);
	Scalar summa = sum(h);
	outputImg = h / summa[0];


	Mat outputImg_norm;
	normalize(outputImg, outputImg_norm, 0, 255, NORM_MINMAX);
	cv::imshow("PSF", outputImg_norm);
}

void fftshift(const Mat& inputImg, Mat& outputImg)
{
	outputImg = inputImg.clone();
	int cx = outputImg.cols / 2;
	int cy = outputImg.rows / 2;
	Mat q0(outputImg, Rect(0, 0, cx, cy));
	Mat q1(outputImg, Rect(cx, 0, cx, cy));
	Mat q2(outputImg, Rect(0, cy, cx, cy));
	Mat q3(outputImg, Rect(cx, cy, cx, cy));
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

void filter2DFreq(const Mat& inputImg, Mat& outputImg, const Mat& H)
{
	Mat planes[2] = { Mat_<double>(inputImg.clone()), Mat::zeros(inputImg.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI, DFT_SCALE);

	Mat planesH[2] = { Mat_<double>(H.clone()), Mat::zeros(H.size(), CV_32F) };
	Mat complexH;
	merge(planesH, 2, complexH);
	Mat complexIH;
	mulSpectrums(complexI, complexH, complexIH, 0);

	idft(complexIH, complexIH);
	split(complexIH, planes);
	outputImg = planes[0];
}

void calcWnrFilter(const Mat& input_h_PSF, Mat& output_G, double nsr)
{
	Mat h_PSF_shifted;
	fftshift(input_h_PSF, h_PSF_shifted);
	Mat planes[2] = { Mat_<double>(h_PSF_shifted.clone()), Mat::zeros(h_PSF_shifted.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);
	dft(complexI, complexI);
	split(complexI, planes);
	Mat denom;
	pow(abs(planes[0]), 2, denom);
	denom += nsr;
	divide(planes[0], denom, output_G);
}

void edgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta)
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



// ========================= ОСНОВНАЯ ФУНКЦИЯ =========================

int main() {
    cout << "========================================" << endl;
    cout << " MULTI-THREADED VIDEO STABILIZER OPENCL " << endl;
    cout << "========================================" << endl;
    
    cv::ocl::setUseOpenCL(USE_OPENCL);
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    
    cout << "Выберите режим работы:" << endl;
    cout << "1. Использовать камеру" << endl;
    cout << "2. Читать кадры из папки PXL_3" << endl;
    cout << "3. Читать кадры из папки PXL_4K" << endl;
    cout << "Введите 1, 2 или 3: ";
    
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
        cout << "Используется режим http камеры по умолчанию:" << "http://192.168.0.103:4747/video?1000x1000" << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "http://192.168.0.103:4747/video?1000x1000";
        }
    }
    else {
        cout << "Используется режим камеры по умолчанию:" << videoSource << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";
        }
    }
    
    VideoStabilizer stabilizer;
    stabilizer.start(useCamera, imageFolderPath);
    
    cout << endl << "Управление:" << endl;
    cout << "  ESC или Q - выход" << endl;
    cout << "  Пробел - пауза" << endl;
    cout << "  F - сохранить текущий кадр" << endl;
    cout << "  D - переключить режим отладки" << endl;
    cout << "  S/W - увеличить/уменьшить область кадра" << endl;
    
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