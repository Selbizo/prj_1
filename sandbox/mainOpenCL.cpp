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
const bool USE_OPENCL_FF = true; // Использовать OpenCL для оптического потока

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
const int compressionConfig = 4;

// Настройки детектора
int maxCornersConfig = 200;
double qualityLevelConfig = 0.005;
const double minDistanceConfig = 3.0;
int blockSizeConfig = 9;
const bool useHarrisDetectorConfig = true;
double harrisKConfig = 0.005;

// Настройки оптического потока
const int winSizeConfig = blockSizeConfig;
const int maxLevelConfig = 5;
const int itersConfig = 10;

// Источник видео
const string videoSource = "http://192.168.0.102:4747/video";

string filepath = "";

// ========================= СТРУКТУРЫ ДАННЫХ =========================

struct TransformParam {
    double dx;
    double dy;
    double da;
    
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
    UMat frame;
    UMat gray;
    vector<Point2f> points;
    TransformParam transformSKO;
    TransformParam transformFirstDerivative;
    TransformParam transform;
    UMat stabMatrix;
    int frameId;
    double timestamp;
    
    FrameData() : frameId(0), timestamp(0) {}
};

class ThreadSafeQueue {
private:
    queue<FrameData> queue_;
    mutable mutex mutex_;
    condition_variable cond_;
    
public:
    void push(FrameData data) {
        {
            lock_guard<mutex> lock(mutex_);
            queue_.push(move(data));
        }
        cond_.notify_one();
    }
    
    bool try_pop(FrameData& data) {
        lock_guard<mutex> lock(mutex_);
        if (queue_.empty()) return false;
        data = move(queue_.front());
        queue_.pop();
        return true;
    }
    
    bool wait_and_pop(FrameData& data) {
        unique_lock<mutex> lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty(); });
        data = move(queue_.front());
        queue_.pop();
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

// Функция для инициализации OpenCL
bool initOpenCL() {
    if (!USE_OPENCL) {
        cout << "OpenCL disabled by configuration" << endl;
        return false;
    }
    
    // Включаем OpenCL в OpenCV
    cv::ocl::setUseOpenCL(true);
    
    // Проверяем, доступен ли OpenCL
    if (!cv::ocl::haveOpenCL()) {
        cerr << "OpenCL is not available on this system" << endl;
        return false;
    }
    
    // Получаем информацию об устройствах OpenCL
    cv::ocl::Context context;
    if (!context.create(cv::ocl::Device::TYPE_GPU)) {
        cerr << "Failed to create OpenCL context for GPU, trying CPU..." << endl;
        if (!context.create(cv::ocl::Device::TYPE_CPU)) {
            cerr << "Failed to create OpenCL context for CPU" << endl;
            return false;
        }
    }
    
    cout << "OpenCL is available and enabled" << endl;
    cout << "Number of OpenCL devices: " << context.ndevices() << endl;
    
    // Выводим информацию о каждом устройстве
    for (size_t i = 0; i < context.ndevices(); i++) {
        cv::ocl::Device device = context.device(i);
        cout << "OpenCL Device " << i << ": " << device.name() << endl;
        cout << "  Version: " << device.version() << endl;
    }
    
    // Используем первое устройство
    cv::ocl::Device(context.device(0));
    
    // Получаем текущее устройство для дополнительной информации
    cv::ocl::Device currentDevice;
    cout << "Current OpenCL device: " << currentDevice.name() << endl;
    //cout << "  Vendor: " << currentDevice.vendor() << endl;
    cout << "  Version: " << currentDevice.version() << endl;
    cout << "  Driver version: " << currentDevice.driverVersion() << endl;
    
    cout << "OpenCV OpenCL status:" << endl;
    cout << "  UMat usage: Enabled" << endl;
    cout << "  OpenCL enabled: " << (cv::ocl::useOpenCL() ? "Yes" : "No") << endl;
    
    return true;
}

// ========================= ОСНОВНЫЕ ФУНКЦИИ =========================

class VideoStabilizer {
private:
    ThreadSafeQueue rawFramesQueue;
    ThreadSafeQueue processedFramesQueue;
    
    unordered_map<int, FrameData> frameBuffer;
    mutex bufferMutex;
    atomic<int> nextDisplayFrameId{0};
    condition_variable frameReadyCV;
    
    mutex resourcesMutex;
    Ptr<GFTTDetector> detector;
    
    UMat prevGray;
    vector<Point2f> prevPoints;
    bool firstFrameForTracking = true;
    
    double tauStab;
    double kSwitch;
    double framePart;
    TransformParam oldTransform;
    Rect roi;
    Size frameSize;
    int a, b;
    
    atomic<int> fps;
    atomic<int> trackedPoints;
    atomic<int> framesProcessed;
    atomic<bool> debugMode;
    
    atomic<int> processingLag{0};
    const int MAX_PROCESSING_LAG = 6;
    const int MAX_BUFFER_SIZE = 5;
    
    double processingTimeCapture;
    double processingTimeDetectionTracking;
    double processingTimeStabilization;
    double processingTimeImshow;

    int captureCore;
    int detectionCore;
    int stabilizationCore;
    int displayCore;

public:
    vector<thread> workers;
    vector<pthread_t> pthreads;
    atomic<bool> running;

    // Специализированные UMat для оптимизации памяти
    UMat grayBuffer;
    UMat compressedBuffer;
    UMat maskBuffer;
    UMat stabilizedBuffer;
    UMat croppedBuffer;

    VideoStabilizer() 
        : running(false), 
          tauStab(100.0), 
          kSwitch(0.1), 
          framePart(0.7),
          trackedPoints(0),
          framesProcessed(0),
          debugMode(false),
          processingTimeCapture(0),
          processingTimeDetectionTracking(0),
          processingTimeStabilization(0),
          processingTimeImshow(0),
          captureCore(-1),
          detectionCore(-1),
          stabilizationCore(-1),
          displayCore(-1) {
        
        int totalCores = getAvailableCores();
        cout << "Available CPU cores: " << totalCores << endl;
        
        // Настройка привязки к ядрам
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
        cout << "Core assignment: Capture->" << captureCore 
             << ", Detection->" << detectionCore 
             << ", Stabilization->" << stabilizationCore 
             << ", Display->" << displayCore << endl;
    }
    
    void stop() {
        running = false;
        
        {
            lock_guard<mutex> lock(bufferMutex);
            frameReadyCV.notify_all();
        }
        
        for (auto& worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers.clear();
        pthreads.clear();
        
        // Очищаем UMat буферы
        grayBuffer.release();
        compressedBuffer.release();
        maskBuffer.release();
        stabilizedBuffer.release();
        croppedBuffer.release();
        
        cout << "Video stabilizer stopped. Total frames processed: " << framesProcessed << endl;
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
        cv::ocl::setUseOpenCL(USE_OPENCL);
        
        if (!useCamera) {
            if (imageFolderPath.empty()) {
                cerr << "File path not specified for image sequence loading" << endl;
                running = false;
                return;
            }
            
            struct stat info;
            if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
                cerr << "Cannot access directory: " << imageFolderPath << endl;
                running = false;
                return;
            }
            
            cout << "Loading image sequence from: " << imageFolderPath << endl;
        }
        
        VideoCapture cap;
        if (useCamera) {
            int cameraIndex = 0;
            if (videoSource == "0") {
                cameraIndex = 0;
            } else {
                try {
                    cameraIndex = stoi(videoSource);
                } catch (...) {
                    cameraIndex = 0;
                }
            }
            
            cap.open(cameraIndex);
            if (!cap.isOpened()) {
                cerr << "Cannot open camera " << cameraIndex << endl;
                running = false;
                return;
            }
            cout << "Opened camera " << cameraIndex << " as video source" << endl;
            
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
            UMat firstFrame;
            loadImage(firstFrame, 0, imageFolderPath);
            
            if (firstFrame.empty()) {
                for (int i = 1; i < 100; i++) {
                    loadImage(firstFrame, i, imageFolderPath);
                    if (!firstFrame.empty()) break;
                }
                
                if (firstFrame.empty()) {
                    cerr << "Cannot find any images in the sequence" << endl;
                    running = false;
                    return;
                }
            }
            
            frameSize = Size(firstFrame.cols, firstFrame.rows);
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
        
        // Предварительно выделяем буферы
        Size compressedSize(a / compressionConfig, b / compressionConfig);
        if (compressedSize.width <= 0) compressedSize.width = 1;
        if (compressedSize.height <= 0) compressedSize.height = 1;
        
        compressedBuffer.create(compressedSize, CV_8UC3);
        grayBuffer.create(compressedSize, CV_8UC1);
        
        while (running) {
            auto startTimeCap = chrono::steady_clock::now();
            int totalLag = rawFramesQueue.size() + processedFramesQueue.size();
            
            if (totalLag > MAX_PROCESSING_LAG) {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            
            FrameData frameData;
            frameData.frameId = frameId++;
            
            bool frameRead = false;
            
            if (useCamera) {
                Mat frame_cpu;
                frameRead = cap.read(frame_cpu);
                if (frameRead) {
                    frame_cpu.copyTo(frameData.frame);
                }
                
                if (!frameRead) {
                    cerr << "Failed to read frame from camera" << endl;
                    continue;
                }
            } else {
                loadImage(frameData.frame, frameData.frameId%1200, imageFolderPath);
                
                if (frameData.frame.empty()) {
                    loadImage(frameData.frame, frameData.frameId%1200 + 1, imageFolderPath);
                    if (frameData.frame.empty()) {
                        cout << "Image sequence ended or no more images available" << endl;
                        running = false;
                        break;
                    }
                    frameData.frameId++;
                }
                
                frameRead = !frameData.frame.empty();
            }
            
            if (!frameRead || frameData.frame.empty()) {
                continue;
            }
            
            if (frameData.frame.cols != a || frameData.frame.rows != b) {
                cout << "Frame size changed from " << a << "x" << b 
                     << " to " << frameData.frame.cols << "x" << frameData.frame.rows << endl;
                a = frameData.frame.cols;
                b = frameData.frame.rows;
                frameSize = Size(a, b);
                
                roi.x = static_cast<int>(a * ((1.0 - framePart) / 2.0));
                roi.y = static_cast<int>(b * ((1.0 - framePart) / 2.0));
                roi.width = static_cast<int>(a * framePart);
                roi.height = static_cast<int>(b * framePart);
                
                roi.x = max(0, min(roi.x, a - roi.width));
                roi.y = max(0, min(roi.y, b - roi.height));
                roi.width = min(roi.width, a - roi.x);
                roi.height = min(roi.height, b - roi.y);
                
                // Обновляем размер буферов
                compressedSize = Size(a / compressionConfig, b / compressionConfig);
                compressedBuffer.create(compressedSize, CV_8UC3);
                grayBuffer.create(compressedSize, CV_8UC1);
            }
            
            // Используем OpenCL для операций с изображениями
            // Сжатие и конвертация в градации серого на GPU
            resize(frameData.frame, compressedBuffer, compressedSize, 0, 0, INTER_CUBIC);
            cvtColor(compressedBuffer, grayBuffer, COLOR_BGR2GRAY);
            grayBuffer.copyTo(frameData.gray);
            
            rawFramesQueue.push(move(frameData));
            
            frameCount++;
            auto now = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFpsTime);
            if (elapsed.count() >= 1000) {
                fps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
                if (debugMode && frameId % 100 == 0) {
                    cout << "Capture FPS: " << fps << ", Queue sizes: " 
                         << rawFramesQueue.size() << "/" << processedFramesQueue.size() << endl;
                }
            }
            
            auto endTimeCap = chrono::steady_clock::now();
            auto durationCap = chrono::duration_cast<chrono::microseconds>(endTimeCap - startTimeCap);
            processingTimeCapture = (processingTimeCapture*19.0 + durationCap.count() / 1000.0)/20.0;
        }
        
        if (useCamera) {
            cap.release();
        }
        
        cout << "Capture thread stopped" << endl;
    }
    
    // ========================= ПОТОК ДЕТЕКТИРОВАНИЯ И ОТСЛЕЖИВАНИЯ =========================
    void detectionAndTrackingThread() {
        cv::ocl::setUseOpenCL(USE_OPENCL);
        cout << "Detection and Tracking thread started" << endl;
        
        TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        Size winSize(winSizeConfig, winSizeConfig);
        int consecutiveFailures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 10;
        
        // Для OpenCL оптического потока
        vector<Point2f> nextPoints;
        vector<uchar> status;
        vector<float> err;
        
        while (running) {
            FrameData frameData;
            if (!rawFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }
            
            auto startTimeDetTrack = chrono::steady_clock::now();
            
            if (firstFrameForTracking || prevPoints.empty() || trackedPoints.load() < maxCornersConfig / 5) {
                UMat mask;
                if (frameData.gray.size().width > 0 && frameData.gray.size().height > 0) {
                    // Создаем маску на GPU
                    mask = UMat::zeros(frameData.gray.size(), CV_8U);
                    int marginX = frameData.gray.cols / 8;
                    int marginY = frameData.gray.rows / 8;
                    
                    // Используем OpenCL для рисования прямоугольника
                    rectangle(mask, 
                            Rect(marginX, marginY, 
                                frameData.gray.cols - 2 * marginX, 
                                frameData.gray.rows - 2 * marginY),
                            Scalar(255), FILLED);
                    
                    vector<KeyPoint> keypoints;
                    try {
                        // Конвертируем UMat в Mat для детектора (детектор не поддерживает UMat)
                        Mat gray_cpu = frameData.gray.getMat(ACCESS_READ);
                        Mat mask_cpu = mask.getMat(ACCESS_READ);
                        detector->detect(gray_cpu, keypoints, mask_cpu);
                    } catch (const exception& e) {
                        cerr << "Error in detector: " << e.what() << endl;
                        continue;
                    }
                    
                    for (const auto& kp : keypoints) {
                        frameData.points.push_back(kp.pt);
                    }
                    
                    if (frameData.points.size() > maxCornersConfig * 4) {
                        frameData.points.erase(frameData.points.begin(), 
                                    frameData.points.begin() + (frameData.points.size() - maxCornersConfig));
                    }
                    
                    removeFramePoints(frameData.points, minDistanceConfig * 0.8);
                }
                
                if (firstFrameForTracking) {
                    frameData.gray.copyTo(prevGray);
                    prevPoints = frameData.points;
                    firstFrameForTracking = false;
                    trackedPoints.store(static_cast<int>(prevPoints.size()));
                    frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                    
                    processedFramesQueue.push(move(frameData));
                    
                    auto endTimeDetTrack = chrono::steady_clock::now();
                    auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
                    processingTimeDetectionTracking = (processingTimeDetectionTracking * 99.0 + durationDetTrack.count() / 1000.0) / 100.0;
                    
                    continue;
                }
            }
            
            if (!firstFrameForTracking && !prevPoints.empty() && !frameData.gray.empty() && !prevGray.empty()) {
                nextPoints.clear();
                status.clear();
                err.clear();
                
                try {
                    if (USE_OPENCL_FF && cv::ocl::useOpenCL()) {
                        // Используем UMat для оптического потока (OpenCL)
                        // OpenCV автоматически использует OpenCL для calcOpticalFlowPyrLK с UMat
                        
                        // Конвертируем vector<Point2f> в UMat
                        UMat prevPtsUMat, nextPtsUMat;
                        Mat(prevPoints).copyTo(prevPtsUMat);
                        
                        // Оптический поток с UMat - будет использовать OpenCL если доступно
                        calcOpticalFlowPyrLK(
                            prevGray,           // UMat
                            frameData.gray,     // UMat
                            prevPtsUMat,        // UMat
                            nextPtsUMat,        // UMat
                            status, err,
                            winSize, maxLevelConfig,
                            termcrit, 0, 0.001
                        );
                        
                        // Конвертируем обратно в vector<Point2f>
                        if (!nextPtsUMat.empty()) {
                            Mat nextPtsMat = nextPtsUMat.getMat(ACCESS_READ);
                            nextPoints.assign((Point2f*)nextPtsMat.data, 
                                             (Point2f*)nextPtsMat.data + nextPtsMat.total());
                        }
                        
                        if (debugMode && frameData.frameId % 30 == 0) {
                            cout << "OpticalFlow using OpenCL: Yes" << endl;
                        }
                    } else {
                        // CPU версия
                        Mat prevGray_cpu = prevGray.getMat(ACCESS_READ);
                        Mat gray_cpu = frameData.gray.getMat(ACCESS_READ);
                        
                        calcOpticalFlowPyrLK(
                            prevGray_cpu, gray_cpu,
                            prevPoints, nextPoints,
                            status, err,
                            winSize, maxLevelConfig,
                            termcrit, 0, 0.001
                        );
                        
                        if (debugMode && frameData.frameId % 30 == 0) {
                            cout << "OpticalFlow using OpenCL: No" << endl;
                        }
                    }
                } catch (const exception& e) {
                    cerr << "Error in optical flow: " << e.what() << endl;
                    frameData.gray.copyTo(prevGray);
                    frameData.points.clear();
                    prevPoints.clear();
                    trackedPoints.store(0);
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
                    Mat T_cpu;
                    try {
                        T_cpu = estimateAffine2D(goodOld, goodNew, noArray(), RANSAC, 3.0);
                    } catch (const exception& e) {
                        cerr << "Error in estimateAffine2D: " << e.what() << endl;
                        T_cpu = Mat();
                    }
                    
                    if (!T_cpu.empty() && T_cpu.rows == 2 && T_cpu.cols == 3) {
                        double dx = T_cpu.at<double>(0, 2) * compressionConfig;
                        double dy = T_cpu.at<double>(1, 2) * compressionConfig;
                        double da = atan2(T_cpu.at<double>(1, 0), 
                                        T_cpu.at<double>(0, 0));
                        
                        frameData.transformFirstDerivative = TransformParam(dx, dy, da);
                        
                        if (debugMode && frameData.frameId % 1 == 0) {
                            frameData.transformFirstDerivative.print();
                        }
                    } else {
                        frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                    }
                } else {
                    frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                }
                
                frameData.gray.copyTo(prevGray);
                prevPoints = goodNew;
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
            
            if ((frameData.points.size() < maxCornersConfig / 6)) {
                maxCornersConfig = 60;
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
            
            processedFramesQueue.push(move(frameData));
        }
        
        cout << "Detection and Tracking thread stopped" << endl;
    }
    
    // ========================= ПОТОК СТАБИЛИЗАЦИИ =========================
    void stabilizationThread() {
        cv::ocl::setUseOpenCL(USE_OPENCL);
        int framesStabilized = 0;
        
        cout << "Stabilization thread started" << endl;
        
        while (running) {
            FrameData frameData;
            if (!processedFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }

            auto startTimeStab = chrono::steady_clock::now();
            
            if (!frameData.frame.empty()) {
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
                    
                    iirAdaptive(frameData.transformFirstDerivative, oldTransform, 
                               frameData.transformSKO, frameData.stabMatrix, tauStab, roi, a, b, kSwitch);
                    
                    frameData.transform = oldTransform;
                    
                    // Рисуем точки на GPU? Не поддерживается напрямую, придется на CPU
                    if (!frameData.points.empty()) {
                        int pointsToShow = min(300, static_cast<int>(frameData.points.size()));
                        Mat frame_cpu = frameData.frame.getMat(ACCESS_WRITE);
                        for (int i = 0; i < pointsToShow; i++) {
                            Point2f pt = frameData.points[i];
                            Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                        static_cast<int>(pt.y * compressionConfig));
                            circle(frame_cpu, scaledPt, 2, Scalar(48, 62, 255), -1);
                        }
                    }

                    // Используем OpenCL для warpAffine
                    if (stabilizedBuffer.empty() || stabilizedBuffer.size() != frameSize) {
                        stabilizedBuffer.create(frameSize, frameData.frame.type());
                    }
                    warpAffine(frameData.frame, stabilizedBuffer, frameData.stabMatrix, frameSize);
                    
                    // Кроппинг на GPU
                    if (croppedBuffer.empty() || croppedBuffer.size() != Size(roi.width, roi.height)) {
                        croppedBuffer.create(Size(roi.width, roi.height), frameData.frame.type());
                    }
                    croppedBuffer = stabilizedBuffer(roi);
                    croppedBuffer.copyTo(frameData.frame);
                    
                    framesStabilized++;
                    
                } catch (const exception& e) {
                    cerr << "Error in stabilization: " << e.what() << endl;
                }
            }

            {
                lock_guard<mutex> lock(bufferMutex);
                frameBuffer[frameData.frameId] = move(frameData);
                frameReadyCV.notify_one();
                
                if (frameBuffer.size() > MAX_BUFFER_SIZE * 2) {
                    vector<int> toRemove;
                    for (auto& pair : frameBuffer) {
                        if (pair.first < nextDisplayFrameId - 5) {
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
            processingTimeStabilization = (processingTimeStabilization*19.0 + durationStab.count() / 1000.0)/20.0;
        }
        
        cout << "Stabilization thread stopped. Stabilized " << framesStabilized 
             << " frames total." << endl;
    }
    
    // ========================= ПОТОК ОТОБРАЖЕНИЯ =========================
    void displayThread() {
        cv::ocl::setUseOpenCL(USE_OPENCL);
        const string windowName = "Video Stabilization";
        namedWindow(windowName, WINDOW_NORMAL);
        resizeWindow(windowName, 1280, 720);
        
        VideoWriter writer;
        if (recordEnable) {
            writer.open("stabilized_output.mp4", 
                       VideoWriter::fourcc('a', 'v', 'c', '1'),
                       30, frameSize);
            if (!writer.isOpened()) {
                cerr << "Failed to open video writer" << endl;
            } else {
                cout << "Video recording enabled" << endl;
            }
        }
        
        int displayedFrames = 0;
        auto lastDisplayTime = chrono::steady_clock::now();
        auto lastFrameTime = chrono::steady_clock::now();
        
        // UMat для отображения
        UMat displayUMat;
        
        cout << "Display thread started" << endl;
        
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
                            
                            if (debugMode) {
                                cout << "Display: skipped to frame " << nextAvailable 
                                     << ", expected " << nextDisplayFrameId - 1 << endl;
                            }
                        }
                    }
                    
                    if (!gotFrame) {
                        frameReadyCV.wait_for(lock, chrono::milliseconds(5));
                        continue;
                    }
                }
            }
            
            if (!gotFrame || frameData.frame.empty()) {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            
            displayedFrames++;
            
            // Используем OpenCL для изменения размера
            if (displayUMat.empty() || displayUMat.size() != Size(a, b)) {
                displayUMat.create(Size(a, b), frameData.frame.type());
            }
            resize(frameData.frame, displayUMat, Size(a, b), 0, 0, INTER_AREA);

            // Конвертируем UMat в Mat для отображения (imshow требует Mat)
            Mat displayFrame_cpu = displayUMat.getMat(ACCESS_READ);
            
            string infoText = format("FPS: %d | Process: %2.1f ms | tauStab: %2.1f | framePart: %1.2f | OpenCL: %s",
                                    fps.load(),
                                    processingTimeCapture + 
                                    processingTimeDetectionTracking + 
                                    processingTimeStabilization + 
                                    processingTimeImshow, 
                                    tauStab, framePart,
                                    cv::ocl::useOpenCL() ? "ON" : "OFF");

            string infoLatencies = format("Capture: %2.1f | Det+Track: %2.1f | Stabilization: %2.1f | Imshow: %2.1f",
                                    processingTimeCapture, processingTimeDetectionTracking, 
                                    processingTimeStabilization, processingTimeImshow);

            putText(displayFrame_cpu, infoText, Point(10, 50*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5*a/800, colorBLUE, 2*a/800);
            putText(displayFrame_cpu, infoLatencies, Point(10, 100*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5*a/800, colorBLUE, 2*a/800);
            
            imshow(windowName, displayFrame_cpu);
            
            auto endTimeDisp = chrono::steady_clock::now();
            auto durationDisp = chrono::duration_cast<chrono::microseconds>(endTimeDisp - startTimeDisp);
            processingTimeImshow = (processingTimeImshow*19.0 + durationDisp.count() / 1000.0)/20.0;

            if (recordEnable && writer.isOpened()) {
                writer.write(displayFrame_cpu);
            }
            
            auto now = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastDisplayTime);
            if (elapsed.count() >= 1000) {
                if (debugMode) {
                    cout << "Display FPS: " << displayedFrames 
                         << ", Buffer size: " << frameBuffer.size() 
                         << ", Next frame: " << nextDisplayFrameId << endl;
                }
                displayedFrames = 0;
                lastDisplayTime = now;
            }
            
            auto frameElapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFrameTime);
            if (frameElapsed.count() < 5) {
                this_thread::sleep_for(chrono::milliseconds(5 - frameElapsed.count()));
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
                imwrite(filename, displayFrame_cpu);
                cout << "Frame saved: " << filename << endl;
            } else if (key == 'd') {
                debugMode = !debugMode;
                cout << "Debug mode: " << (debugMode ? "ON" : "OFF") << endl;
            } else if (key == 'o') {
                // Переключение OpenCL
                bool current = cv::ocl::useOpenCL();
                cv::ocl::setUseOpenCL(!current);
                cout << "OpenCL " << (cv::ocl::useOpenCL() ? "enabled" : "disabled") << endl;
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
        
        if (writer.isOpened()) {
            writer.release();
            cout << "Video writer released" << endl;
        }
        destroyWindow(windowName);
        cout << "Display thread stopped" << endl;
    }
    
    // ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================
    void loadImage(UMat& image_color, int frame_id, std::string filepath) {
        cv::ocl::setUseOpenCL(USE_OPENCL);
        char file[200];
        std::string filename;
        
        sprintf(file, "image_0/%06d.png", frame_id);
        filename = filepath + std::string(file);
        
        std::ifstream file_check(filename.c_str());
        if (file_check.good()) {
            file_check.close();
            Mat image_cpu = imread(filename, IMREAD_COLOR);
            if (!image_cpu.empty()) {
                image_cpu.copyTo(image_color);
            }
        }
        
        if (image_color.empty()) {
            sprintf(file, "image_0/%06d.jpg", frame_id);
            filename = filepath + std::string(file);
            
            std::ifstream jpg_check(filename.c_str());
            if (jpg_check.good()) {
                jpg_check.close();
                Mat image_cpu = imread(filename, IMREAD_COLOR);
                if (!image_cpu.empty()) {
                    image_cpu.copyTo(image_color);
                }
            }
        }
    }

    void removeFramePoints(vector<Point2f>& p0, double minDistance) {
        if (p0.empty()) return;

        std::sort(p0.begin(), p0.end(), [](const cv::Point2f& a, const cv::Point2f& b) {
            return a.x < b.x;
        });

        std::vector<bool> toRemove(p0.size(), false);
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

        if (tauStab < 30.0) tauStab *= 1.2;

        if (tauStab < 50.0 && !(abs(transforms.dx) > a / 2 || abs(transforms.dy) > b / 2))
            tauStab *= 1.1;

        if (tauStab < 100.0 && !(abs(transforms.dx) > a / 3 || abs(transforms.dy) > b / 3)) {
            tauStab *= 1.1;
            if (tauStab > 100.0) tauStab = 100.0;
        }

        if (roi.x + (int)transforms.dx < 0) {
            transforms.dx = double(1 - roi.x);
            if (tauStab > 50) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        } else if (roi.x + roi.width + (int)transforms.dx >= a) {
            transforms.dx = (double)(a - roi.x - roi.width);
            if (tauStab > 50) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (roi.y + (int)transforms.dy < 0) {
            transforms.dy = (double)(1 - roi.y);
            if (tauStab > 10) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        } else if (roi.y + roi.height + (int)transforms.dy >= b) {
            transforms.dy = (double)(b - roi.y - roi.height);
            if (tauStab > 50) {
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

// ========================= ОСНОВНАЯ ФУНКЦИЯ =========================

int main() {
    cout << "========================================" << endl;
    cout << " MULTI-THREADED VIDEO STABILIZER OPENCL " << endl;
    cout << "========================================" << endl;
    
    // Инициализация OpenCL
    bool openclAvailable = initOpenCL();
    if (openclAvailable) {
        cout << "OpenCL successfully initialized" << endl;
    } else {
        cout << "Running in CPU-only mode" << endl;
    }
    
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    
    cout << "Выберите режим работы:" << endl;
    cout << "1. Использовать камеру (по умолчанию)" << endl;
    cout << "3. Читать кадры из папки PXL_3" << endl;
    cout << "4. Читать кадры из папки PXL_4K" << endl;
    cout << "5. Читать кадры из папки PXL_4K (x86 PC)" << endl;
    cout << "Введите 1, 3, 4 или 5: ";
    
    int choice;
    cin >> choice;
    
    bool useCamera = true;
    string imageFolderPath;
    
    if (choice == 4) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/pi/opencv_projects/videos/PXL_4K/" << endl;
        cout << "Путь: ";
        
        cin.ignore();
        getline(cin, imageFolderPath);
        
        if (!imageFolderPath.empty()) {
            if (imageFolderPath.front() == '"' || imageFolderPath.front() == '\'') {
                imageFolderPath.erase(0, 1);
            }
            if (imageFolderPath.back() == '"' || imageFolderPath.back() == '\'') {
                imageFolderPath.pop_back();
            }
        }
        
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        cout << "Путь к кадрам: " << imageFolderPath << endl;
        
        struct stat info;
        if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            cerr << "Ошибка: директория не существует или недоступна!\n Использование директории по умолчанию." << endl;
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_4K/";
        }
    } else if (choice > 4) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/selbizo/CV/dataset/videos/PXL_4K/" << endl;
        cout << "Путь: ";
        
        cin.ignore();
        getline(cin, imageFolderPath);
        
        if (!imageFolderPath.empty()) {
            if (imageFolderPath.front() == '"' || imageFolderPath.front() == '\'') {
                imageFolderPath.erase(0, 1);
            }
            if (imageFolderPath.back() == '"' || imageFolderPath.back() == '\'') {
                imageFolderPath.pop_back();
            }
        }
        
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        cout << "Путь к кадрам: " << imageFolderPath << endl;
        
        struct stat info;
        if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            cerr << "Ошибка: директория не существует или недоступна!\n Использование директории по умолчанию." << endl;
            imageFolderPath = "/home/selbizo/CV/dataset/videos/PXL_4K/";
        }
    } else if (choice == 3 || choice == 2) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/pi/opencv_projects/videos/PXL_3/" << endl;
        cout << "Путь: ";
        
        cin.ignore();
        getline(cin, imageFolderPath);
        
        if (!imageFolderPath.empty()) {
            if (imageFolderPath.front() == '"' || imageFolderPath.front() == '\'') {
                imageFolderPath.erase(0, 1);
            }
            if (imageFolderPath.back() == '"' || imageFolderPath.back() == '\'') {
                imageFolderPath.pop_back();
            }
        }
        
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        cout << "Путь к кадрам: " << imageFolderPath << endl;
        
        struct stat info;
        if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            cerr << "Ошибка: директория не существует или недоступна!\n Использование директории по умолчанию." << endl;
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_3/";
        }
    } else {
        cout << "Используется режим камеры" << endl;
    }
    
    VideoStabilizer stabilizer;
    stabilizer.start(useCamera, imageFolderPath);
    
    cout << endl << "Управление:" << endl;
    cout << "  ESC или Q - выход" << endl;
    cout << "  Пробел - пауза" << endl;
    cout << "  F - сохранить текущий кадр" << endl;
    cout << "  D - переключить режим отладки" << endl;
    cout << "  O - переключить OpenCL" << endl;
    cout << "  S/W - увеличить/уменьшить область кадра" << endl;
    
    try {
        while (stabilizer.running) {
            this_thread::sleep_for(chrono::seconds(20));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}