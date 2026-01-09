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
const float DEG_TO_RAD = static_cast<float>(CV_PI) / 180.0f;
const float RAD_TO_DEG = 180.0f / static_cast<float>(CV_PI);

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
const int compressionConfig = 4; // Сжатие для обработки

// Настройки детектора
int maxCornersConfig = 200;
float qualityLevelConfig = 0.005f;
const float minDistanceConfig = 3.0f;
int blockSizeConfig = 9;
const bool useHarrisDetectorConfig = true;
float harrisKConfig = 0.005f;

// Настройки оптического потока
const int winSizeConfig = blockSizeConfig;
const int maxLevelConfig = 5;
const int itersConfig = 10;

// Источник видео (измените на свой)
const string videoSource = "http://192.168.0.102:4747/video";

// ========================= СТРУКТУРЫ ДАННЫХ =========================

struct TransformParam {
    float dx;
    float dy;
    float da; // угол в радианах
    
    TransformParam() : dx(0.0f), dy(0.0f), da(0.0f) {}
    TransformParam(float _dx, float _dy, float _da) 
        : dx(_dx), dy(_dy), da(_da) {}
    
    void getTransform(Mat& T) const {
        T = Mat::zeros(2, 3, CV_32F);
        T.at<float>(0, 0) = cosf(da);
        T.at<float>(0, 1) = -sinf(da);
        T.at<float>(0, 2) = dx;
        T.at<float>(1, 0) = sinf(da);
        T.at<float>(1, 1) = cosf(da);
        T.at<float>(1, 2) = dy;
    }
    
    void getTransformInvert(Mat& T) const {
        T = Mat::zeros(2, 3, CV_32F);
        T.at<float>(0, 0) = cosf(da);
        T.at<float>(0, 1) = sinf(da);
        T.at<float>(0, 2) = -dx;
        T.at<float>(1, 0) = -sinf(da);
        T.at<float>(1, 1) = cosf(da);
        T.at<float>(1, 2) = -dy;
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
    Mat frame;
    Mat gray;
    vector<Point2f> points;
    TransformParam transformSKO;
    TransformParam transformFirstDerivative;
    TransformParam transform;
    Mat stabMatrix;
    int frameId;
    float timestamp;
    
    FrameData() : frameId(0), timestamp(0.0f) {}
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

// ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ПРИВЯЗКИ ЯДЕР =========================

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
    ThreadSafeQueue rawFramesQueue;
    ThreadSafeQueue processedFramesQueue;
    
    unordered_map<int, FrameData> frameBuffer;
    mutex bufferMutex;
    atomic<int> nextDisplayFrameId{0};
    condition_variable frameReadyCV;
    
    mutex resourcesMutex;
    Ptr<GFTTDetector> detector;
    
    Mat prevGray;
    vector<Point2f> prevPoints;
    bool firstFrameForTracking = true;
    
    float tauStab;
    float kSwitch;
    float framePart;
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
    
    float processingTimeCapture;
    float processingTimeDetectionTracking;
    float processingTimeStabilization;
    float processingTimeImshow;

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
          tauStab(100.0f), 
          kSwitch(0.1f), 
          framePart(0.7f),
          trackedPoints(0),
          framesProcessed(0),
          debugMode(false),
          processingTimeCapture(0.0f),
          processingTimeDetectionTracking(0.0f),
          processingTimeStabilization(0.0f),
          processingTimeImshow(0.0f),
          captureCore(-1),
          detectionCore(-1),
          stabilizationCore(-1),
          displayCore(-1) {
        
        int totalCores = getAvailableCores();
        cout << "Available CPU cores: " << totalCores << endl;
        
        if (totalCores >= 6) {
            captureCore = 2;
            detectionCore = 3;
            stabilizationCore = 4;
            displayCore = 5;
        } else if (totalCores >= 4) {
            captureCore = 0;
            detectionCore = 1;
            stabilizationCore = 2;
            displayCore = 3;
        } else {
            captureCore = -1;
            detectionCore = -1;
            stabilizationCore = -1;
            displayCore = -1;
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
        
        if (captureCore >= 0) {
            setThreadAffinity(pthreads[0], captureCore);
        }
        if (detectionCore >= 0) {
            setThreadAffinity(pthreads[1], detectionCore);
        }
        if (stabilizationCore >= 0) {
            setThreadAffinity(pthreads[2], stabilizationCore);
        }
        if (displayCore >= 0) {
            setThreadAffinity(pthreads[3], displayCore);
        }
        
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
        cout << "Video stabilizer stopped. Total frames processed: " << framesProcessed << endl;
    }
    
private:
    void captureThreadWrapper(bool useCamera, const string& imageFolderPath) {
        if (captureCore >= 0) {
            setCurrentThreadAffinity(captureCore);
        }
        captureThread(useCamera, imageFolderPath);
    }
    
    void detectionAndTrackingThreadWrapper() {
        if (detectionCore >= 0) {
            setCurrentThreadAffinity(detectionCore);
        }
        detectionAndTrackingThread();
    }
    
    void stabilizationThreadWrapper() {
        if (stabilizationCore >= 0) {
            setCurrentThreadAffinity(stabilizationCore);
        }
        stabilizationThread();
    }
    
    void displayThreadWrapper() {
        if (displayCore >= 0) {
            setCurrentThreadAffinity(displayCore);
        }
        displayThread();
    }
    
    void captureThread(bool useCamera, const string& imageFolderPath) {
        if (!useCamera) {
            if (imageFolderPath.empty()) {
                cerr << "File path not specified for image sequence loading" << endl;
                running = false;
                return;
            }
            
            #if __cplusplus >= 201703L
            error_code ec;
            if (!filesystem::exists(imageFolderPath, ec) || !filesystem::is_directory(imageFolderPath, ec)) {
                cerr << "Cannot access directory: " << imageFolderPath << endl;
                running = false;
                return;
            }
            #else
            struct stat info;
            if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
                cerr << "Cannot access directory: " << imageFolderPath << endl;
                running = false;
                return;
            }
            #endif
            
            cout << "Loading image sequence from: " << imageFolderPath << endl;
        }
        
        VideoCapture cap;
        Mat frame, compressed, gray;
        FrameData frameData;
        Size compressedSize;
        
        if (useCamera) {
            int cameraIndex = (videoSource == "0") ? 0 : stoi(videoSource, nullptr, 10);
            
            cap.open(cameraIndex, CAP_ANY);
            if (!cap.isOpened()) {
                cap.open(cameraIndex);
            }
            
            if (!cap.isOpened()) {
                cerr << "Cannot open camera " << cameraIndex << endl;
                running = false;
                return;
            }
            
            cout << "Opened camera " << cameraIndex << " as video source" << endl;
            
            cap.set(CAP_PROP_BUFFERSIZE, 1);
            cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
            
            int width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));
            int height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT));
            
            if (width <= 0 || height <= 0) {
                width = 640; height = 480;
                cap.set(CAP_PROP_FRAME_WIDTH, width);
                cap.set(CAP_PROP_FRAME_HEIGHT, height);
            }
            
            frameSize = Size(width, height);
            a = width;
            b = height;
        } else {
            Mat firstFrame;
            bool found = false;
            
            for (int i = 0; i < 10 && !found; ++i) {
                loadImage(firstFrame, i, imageFolderPath);
                if (!firstFrame.empty()) {
                    found = true;
                    break;
                }
            }
            
            if (!found) {
                cerr << "Cannot find any images in the sequence" << endl;
                running = false;
                return;
            }
            
            frameSize = Size(firstFrame.cols, firstFrame.rows);
            a = frameSize.width;
            b = frameSize.height;
        }
        
        const float roiOffset = (1.0f - framePart) * 0.5f;
        const int roiX = static_cast<int>(static_cast<float>(a) * roiOffset);
        const int roiY = static_cast<int>(static_cast<float>(b) * roiOffset);
        const int roiWidth = static_cast<int>(static_cast<float>(a) * framePart);
        const int roiHeight = static_cast<int>(static_cast<float>(b) * framePart);
        
        roi.x = max(0, min(roiX, a - roiWidth));
        roi.y = max(0, min(roiY, b - roiHeight));
        roi.width = min(roiWidth, a - roi.x);
        roi.height = min(roiHeight, b - roi.y);
        
        cout << "Resolution: " << a << "x" << b << endl;
        cout << "ROI: x=" << roi.x << " y=" << roi.y << " w=" << roi.width << " h=" << roi.height << endl;
        
        int frameId = 0;
        auto lastFpsTime = chrono::steady_clock::now();
        int frameCount = 0;
        
        compressedSize = Size(max(1, a / compressionConfig), max(1, b / compressionConfig));
        
        while (running) {
            if (rawFramesQueue.size() + processedFramesQueue.size() > MAX_PROCESSING_LAG) {
                this_thread::yield();
                continue;
            }
            auto startTimeCap = chrono::steady_clock::now();
            frameData.frameId = frameId++;
            
            bool frameRead = false;
            
            if (useCamera) {
                if (cap.grab()) {
                    cap.retrieve(frame);
                    frameRead = !frame.empty();
                }
            } else {
                int imageIndex = frameData.frameId % 550;
                loadImage(frameData.frame, imageIndex, imageFolderPath);
                
                if (frameData.frame.empty()) {
                    loadImage(frameData.frame, imageIndex + 1, imageFolderPath);
                    if (frameData.frame.empty()) {
                        cout << "Image sequence ended" << endl;
                        running = false;
                        break;
                    }
                    ++frameData.frameId;
                }
                
                frame = frameData.frame;
                frameRead = true;
            }
            
            if (!frameRead || frame.empty()) {
                if (useCamera) {
                    cerr << "Empty frame captured" << endl;
                }
                continue;
            }
            
            if (frame.cols != a || frame.rows != b) {
                a = frame.cols;
                b = frame.rows;
                frameSize = Size(a, b);
                
                const int newRoiX = static_cast<int>(static_cast<float>(a) * roiOffset);
                const int newRoiY = static_cast<int>(static_cast<float>(b) * roiOffset);
                const int newRoiWidth = static_cast<int>(static_cast<float>(a) * framePart);
                const int newRoiHeight = static_cast<int>(static_cast<float>(b) * framePart);
                
                roi.x = max(0, min(newRoiX, a - newRoiWidth));
                roi.y = max(0, min(newRoiY, b - newRoiHeight));
                roi.width = min(newRoiWidth, a - roi.x);
                roi.height = min(newRoiHeight, b - roi.y);
                
                compressedSize = Size(max(1, a / compressionConfig), max(1, b / compressionConfig));
                
                cout << "Frame size changed to " << a << "x" << b << endl;
            }
            
            if (useCamera) {
                frame.copyTo(frameData.frame);
            }
            
            resize(frameData.frame, compressed, compressedSize, 0, 0, INTER_AREA);
            cvtColor(compressed, gray, COLOR_BGR2GRAY);
            frameData.gray = gray;
            
            rawFramesQueue.push(move(frameData));
            
            if (++frameCount >= 50) {
                auto now = chrono::steady_clock::now();
                auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFpsTime);
                
                if (elapsed.count() >= 1000) {
                    fps = static_cast<int>(frameCount * 1000.0f / static_cast<float>(elapsed.count()));
                    frameCount = 0;
                    lastFpsTime = now;
                    
                    if (debugMode) {
                        cout << "Capture FPS: " << fps << ", Queues: " 
                            << rawFramesQueue.size() << "/" << processedFramesQueue.size() << endl;
                    }
                }
            }
            auto endTimeCap = chrono::steady_clock::now();
            auto durationCap = chrono::duration_cast<chrono::microseconds>(endTimeCap - startTimeCap);
            processingTimeCapture = (processingTimeCapture * 199.0f + static_cast<float>(durationCap.count()) / 1000.0f) / 200.0f;

        }
        
        if (useCamera) {
            cap.release();
        }
        
        cout << "Capture thread stopped" << endl;
    }
    
    void detectionAndTrackingThread() {
        cout << "Detection and Tracking thread started" << endl;
        
        TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03f);
        Size winSize(winSizeConfig, winSizeConfig);
        int consecutiveFailures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 10;
        
        while (running) {
            FrameData frameData;
            if (!rawFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }
            
            auto startTimeDetTrack = chrono::steady_clock::now();
            
            if (firstFrameForTracking || prevPoints.empty() || trackedPoints.load() < maxCornersConfig / 5) {
                Mat mask = Mat::zeros(frameData.gray.size(), CV_8U);
                int marginX = frameData.gray.cols / 8;
                int marginY = frameData.gray.rows / 8;
                rectangle(mask, 
                        Rect(marginX, marginY, 
                            frameData.gray.cols - 2 * marginX, 
                            frameData.gray.rows - 2 * marginY),
                        Scalar(255), FILLED);
                
                vector<KeyPoint> keypoints;
                try {
                    detector->detect(frameData.gray, keypoints, mask);
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
                
                removeFramePoints(frameData.points, minDistanceConfig * 0.8f);
                
                if (firstFrameForTracking) {
                    frameData.gray.copyTo(prevGray);
                    prevPoints = frameData.points;
                    firstFrameForTracking = false;
                    trackedPoints.store(static_cast<int>(prevPoints.size()));
                    frameData.transformFirstDerivative = TransformParam(0.0f, 0.0f, 0.0f);
                    
                    processedFramesQueue.push(move(frameData));
                    
                    auto endTimeDetTrack = chrono::steady_clock::now();
                    auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
                    processingTimeDetectionTracking = (processingTimeDetectionTracking * 99.0f + static_cast<float>(durationDetTrack.count()) / 1000.0f) / 100.0f;
                    
                    continue;
                }
            }
            
            if (!firstFrameForTracking && !prevPoints.empty() && !frameData.gray.empty() && !prevGray.empty()) {
                vector<Point2f> nextPoints;
                vector<uchar> status;
                vector<float> err;
                
                try {
                    calcOpticalFlowPyrLK(
                        prevGray, frameData.gray,
                        prevPoints, nextPoints,
                        status, err,
                        winSize, maxLevelConfig,
                        termcrit, 0, 0.001f
                    );
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
                    if (status[i] && err[i] < 50.0f) {
                        goodNew.push_back(nextPoints[i]);
                        goodOld.push_back(prevPoints[i]);
                        goodCount++;
                    }
                }
                
                if (goodCount >= 6) {
                    Mat T;
                    try {
                        T = estimateAffine2D(goodOld, goodNew, noArray(), RANSAC, 3.0f);
                    } catch (const exception& e) {
                        cerr << "Error in estimateAffine2D: " << e.what() << endl;
                        T = Mat();
                    }
                    
                    if (!T.empty() && T.rows == 2 && T.cols == 3) {
                        // Убедимся, что матрица имеет правильный тип CV_32F
                        Mat T_float;
                        T.convertTo(T_float, CV_32F);
                        
                        float dx = T_float.at<float>(0, 2) * static_cast<float>(compressionConfig);
                        float dy = T_float.at<float>(1, 2) * static_cast<float>(compressionConfig);
                        float da = atan2f(T_float.at<float>(1, 0), T_float.at<float>(0, 0));
                        
                        frameData.transformFirstDerivative = TransformParam(dx, dy, da);
                        
                        if (debugMode && frameData.frameId % 1 == 0) {
                            frameData.transformFirstDerivative.print();
                        }
                    } else {
                        frameData.transformFirstDerivative = TransformParam(0.0f, 0.0f, 0.0f);
                    }
                } else {
                    frameData.transformFirstDerivative = TransformParam(0.0f, 0.0f, 0.0f);
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
                frameData.transformFirstDerivative = TransformParam(0.0f, 0.0f, 0.0f);
            }
            
            if ((frameData.points.size() < maxCornersConfig / 6)) {
                maxCornersConfig = 60;
                qualityLevelConfig *= 0.98f;
                harrisKConfig *= 0.98f;
                detector->setQualityLevel(qualityLevelConfig);
                detector->setK(harrisKConfig);
            }

            if ((frameData.points.size() > maxCornersConfig * 4 / 5)) {
                qualityLevelConfig *= 1.02f;
                harrisKConfig *= 1.02f;
                detector->setQualityLevel(qualityLevelConfig);
                detector->setK(harrisKConfig);
            }

            auto endTimeDetTrack = chrono::steady_clock::now();
            auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
            processingTimeDetectionTracking = (processingTimeDetectionTracking * 99.0f + static_cast<float>(durationDetTrack.count()) / 1000.0f) / 100.0f;
            
            processedFramesQueue.push(move(frameData));
        }
        
        cout << "Detection and Tracking thread stopped" << endl;
    }
    
    void stabilizationThread() {
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
                    if (kSwitch < 0.01f) 
                        kSwitch = 0.01f;
                    if (kSwitch < 1.0f)
                    {
                        kSwitch *= 1.06f;
                        kSwitch += 0.005f;
                    }
                    else if (kSwitch > 1.0f)
                        kSwitch = 1.0f;
                    

                    kSwitch = 1.0f;
                    framesProcessed++;
                    
                    iirAdaptive(frameData.transformFirstDerivative, oldTransform, 
                               frameData.transformSKO, frameData.stabMatrix, tauStab, roi, a, b, kSwitch);
                    
                    frameData.transform = oldTransform;
                    
                    Mat stabilizedFrame, croppedFrame;

                    if (!frameData.points.empty()) {
                        int pointsToShow = min(300, static_cast<int>(frameData.points.size()));
                        for (int i = 0; i < pointsToShow; i++) {
                            Point2f pt = frameData.points[i];
                            Point scaledPt(static_cast<int>(pt.x * static_cast<float>(compressionConfig)),
                                        static_cast<int>(pt.y * static_cast<float>(compressionConfig)));
                            circle(frameData.frame, scaledPt, 2, colorRED, -1);
                        }
                    }

                    warpAffine(frameData.frame, stabilizedFrame, frameData.stabMatrix, frameSize, INTER_LINEAR);
                    
                    croppedFrame = stabilizedFrame(roi);
                    croppedFrame.copyTo(frameData.frame);
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
            processingTimeStabilization = (processingTimeStabilization * 199.0f + static_cast<float>(durationStab.count()) / 1000.0f) / 200.0f;
        }
        
        cout << "Stabilization thread stopped. Stabilized " << framesStabilized 
             << " frames total." << endl;
    }
    
    void displayThread() {
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
            
            string infoText = format("FPS: %d | Process: %2.1f ms | tauStab: %2.1f, | framePart: %1.2f",
                                    fps.load(),
                                    processingTimeCapture + 
                                    processingTimeDetectionTracking + 
                                    processingTimeStabilization + 
                                    processingTimeImshow, 
                                    tauStab, framePart);

            string infoLatencies = format("Capture: %2.1f | Det+Track: %2.1f | Stabilization: %2.1f | Imshow: %2.1f",
                                    processingTimeCapture, processingTimeDetectionTracking, 
                                    processingTimeStabilization, processingTimeImshow);

            putText(frameData.frame, infoText, Point(10, 50*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5f*a/800, colorBLUE, 2*a/800);
            putText(frameData.frame, infoLatencies, Point(10, 100*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5f*a/800, colorBLUE, 2*a/800);
                        
            imshow(windowName, frameData.frame);
            
            auto endTimeDisp = chrono::steady_clock::now();
            auto durationDisp = chrono::duration_cast<chrono::microseconds>(endTimeDisp - startTimeDisp);
            processingTimeImshow = (processingTimeImshow * 199.0f + static_cast<float>(durationDisp.count()) / 1000.0f) / 200.0f;

            if (recordEnable && writer.isOpened()) {
                writer.write(frameData.frame);
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
                imwrite(filename, frameData.frame);
                cout << "Frame saved: " << filename << endl;
            } else if (key == 'd') {
                debugMode = !debugMode;
                cout << "Debug mode: " << (debugMode ? "ON" : "OFF") << endl;
            } else if (key == 's' || key == 'S') {
                if (framePart < 0.95f)
                {
                    framePart *= 1.01f;
                    if (framePart > 0.9f)
                        framePart = 0.9f;
                    roi.x = static_cast<int>(static_cast<float>(a) * ((1.0f - framePart) / 2.0f));
                    roi.y = static_cast<int>(static_cast<float>(b) * ((1.0f - framePart) / 2.0f));
                    roi.width = static_cast<int>(static_cast<float>(a) * framePart);
                    roi.height = static_cast<int>(static_cast<float>(b) * framePart);
                }
            }
            if (key == 'w' || key == 'W')
            {
                if (framePart > 0.2f)
                {
                    framePart *= 0.99f;
                    if (framePart < 0.05f)
                        framePart = 0.05f;
                    roi.x = static_cast<int>(static_cast<float>(a) * ((1.0f - framePart) / 2.0f));
                    roi.y = static_cast<int>(static_cast<float>(b) * ((1.0f - framePart) / 2.0f));
                    roi.width = static_cast<int>(static_cast<float>(a) * framePart);
                    roi.height = static_cast<int>(static_cast<float>(b) * framePart);
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
    
    void loadImage(cv::Mat& image_color, int frame_id, std::string filepath)
    {
        char file[200];
        std::string filename;
        
        sprintf(file, "image_0/%06d.png", frame_id);
        filename = filepath + std::string(file);
        
        std::ifstream file_check(filename.c_str());
        if (file_check.good())
        {
            file_check.close();
            image_color = cv::imread(filename, cv::IMREAD_COLOR);
        }
        
        if (image_color.empty())
        {
            sprintf(file, "image_0/%06d.jpg", frame_id);
            filename = filepath + std::string(file);
            
            std::ifstream jpg_check(filename.c_str());
            if (jpg_check.good())
            {
                jpg_check.close();
                image_color = cv::imread(filename, cv::IMREAD_COLOR);
            }
        }
    }

    void removeFramePoints(vector<Point2f>& p0, float minDistance)
    {
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

        for (int i = static_cast<int>(p0.size()) - 1; i >= 0; --i) {
            if (toRemove[i]) {
                p0.erase(p0.begin() + i);
            }
        }
    }

    void iirAdaptive(TransformParam& transformsFirtsDerivative, TransformParam& transforms, TransformParam& transformSKO, Mat& stabMatrix, 
        float& tauStab, Rect& roi, const int a, const int b, float& kSwitch)
    {
        if (fabsf(transformsFirtsDerivative.dx) < 4.0f * transformSKO.dx + static_cast<float>(a)/8.0f)
        {
            transforms.dx = kSwitch * (transforms.dx * (tauStab - 1.0f) / tauStab + kSwitch * transformsFirtsDerivative.dx);
        } 

        if (fabsf(transformsFirtsDerivative.dy) < 4.0f * transformSKO.dy + static_cast<float>(b)/8.0f)
        {
            transforms.dy = kSwitch * (transforms.dy * (tauStab - 1.0f) / tauStab + kSwitch * transformsFirtsDerivative.dy);
        }

        if (fabsf(transformsFirtsDerivative.da) < 4.0f * transformSKO.da + 0.1f)
        {
            transforms.da = kSwitch * (transforms.da * (tauStab - 1.0f) / tauStab + kSwitch * transformsFirtsDerivative.da);
        }

        if (transforms.da > static_cast<float>(CV_PI))
            transforms.da -= static_cast<float>(CV_PI);
        if (transforms.da < -static_cast<float>(CV_PI))
            transforms.da += static_cast<float>(CV_PI);

        if (tauStab < 30.0f)
            tauStab *= 1.2f;

        if (tauStab < 50.0f && !(fabsf(transforms.dx) > static_cast<float>(a) / 2.0f || fabsf(transforms.dy) > static_cast<float>(b) / 2.0f))
            tauStab *= 1.1f;

        if (tauStab < 100.0f && !(fabsf(transforms.dx) > static_cast<float>(a) / 3.0f || fabsf(transforms.dy) > static_cast<float>(b) / 3.0f))
        {
            tauStab *= 1.1f;
            if (tauStab > 100.0f)
                tauStab = 100.0f;
        }

        if (roi.x + static_cast<int>(transforms.dx) < 0)
        {
            transforms.dx = static_cast<float>(1 - roi.x);
            if (tauStab > 50.0f) {
                tauStab *= 0.9f;
                kSwitch *= 0.95f;
            }
        }
        else if (roi.x + roi.width + static_cast<int>(transforms.dx) >= a)
        {
            transforms.dx = static_cast<float>(a - roi.x - roi.width);
            if (tauStab > 50.0f) {
                tauStab *= 0.9f;
                kSwitch *= 0.95f;
            }
        }

        if (roi.y + static_cast<int>(transforms.dy) < 0)
        {
            transforms.dy = static_cast<float>(1 - roi.y);
            if (tauStab > 10.0f) {
                tauStab *= 0.9f;
                kSwitch *= 0.95f;
            }
        }
        else if (roi.y + roi.height + static_cast<int>(transforms.dy) >= b)
        {
            transforms.dy = static_cast<float>(b - roi.y - roi.height);
            if (tauStab > 50.0f) {
                tauStab *= 0.9f;
                kSwitch *= 0.95f;
            }
        }

        if (kSwitch < 1.0f)
            tauStab *= (4.0f + kSwitch) / 5.0f;

        transformSKO.dx = (1.0f - 0.1f) * transformSKO.dx + 0.1f * fabsf(transformsFirtsDerivative.dx);
        transformSKO.dy = (1.0f - 0.1f) * transformSKO.dy + 0.1f * fabsf(transformsFirtsDerivative.dy);
        transformSKO.da = (1.0f - 0.1f) * transformSKO.da + 0.1f * fabsf(transformsFirtsDerivative.da);

        transforms.getTransformInvert(stabMatrix);
    }
};

int main() {
    cout << "========================================" << endl;
    cout << "     MULTI-THREADED VIDEO STABILIZER    " << endl;
    cout << "========================================" << endl;
    
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    cout << "CPU architecture: Amlogic A311D (4x Cortex-A73 + 2x Cortex-A53)" << endl;
    
    cout << "Выберите режим работы:" << endl;
    cout << "1. Использовать камеру (по умолчанию)" << endl;
    cout << "3. Читать кадры из папки PXL_3 (по умолчанию)"<< endl;
    cout << "4. Читать кадры из папки PXL_4K (по умолчанию)"<< endl;
    cout << "Введите 1, 3 или 4: ";
    
    int choice;
    cin >> choice;
    
    bool useCamera = true;
    string imageFolderPath;
    
    if (choice > 3) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/bananapi/Opencv_projects/dataset/videos/PXL_4K/" << endl;
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
            imageFolderPath = "/home/bananapi/Opencv_projects/dataset/videos/PXL_4K/";
        }
    } else if (choice == 3 || choice == 2) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/bananapi/Opencv_projects/dataset/videos/PXL_3/" << endl;
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
            imageFolderPath = "/home/bananapi/Opencv_projects/dataset/videos/PXL_3/";
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
    cout << "  S/W - увеличить/уменьшить область кадра" << endl;
    
    try {
        while (stabilizer.running)
        {
            this_thread::sleep_for(chrono::seconds(10));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}