#include "stabilizer/video_stabilizer.h"
#include "sync/thread_safe_queue.h"
#include "utils/helpers.h"
#include "gpu/warp_affine.h"
#include "config/config.h"
#include "config/structures.h"

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
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

using namespace cv;
using namespace std;
namespace fs = filesystem;

// ========================= ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ =========================
int maxCornersConfig = 200;
double qualityLevelConfig = 0.005;
int blockSizeConfig = 9;
double harrisKConfig = 0.005;
string videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";
double nsr = 0.01;
double LEN = 0;
double THETA = 0.0;
double D = 0.2;
double TRUE_LEN = 0;
double TRUE_THETA = 0.0;
bool wiener = true;
bool threadwiener = false;
double framePart = 0.7;
Rect roi;
int frameWidth = 0, frameHeight = 0;
int a = 0, b = 0;

// Глобальные OpenCL переменные для Винеровского фильтра
// Инициализируются лениво через getGloalGw() чтобы избежать сегфолта при статической инициализации (OpenCL ещё не готов)
cv::UMat& getGlobalGw() {
    static cv::UMat inst(cv::Size(0, 0), CV_32F);
    return inst;
}
cv::UMat& getGlobalG() {
    static cv::UMat inst(cv::Size(0, 0), CV_32F);
    return inst;
}
cv::UMat& getGlobalGGrayWiener() {
    static cv::UMat inst(cv::Size(0, 0), CV_32F);
    return inst;
}
#define gHw getGlobalGw()
#define gH  getGlobalG()
#define gGrayWiener getGlobalGGrayWiener()

// ========================= КОНСТРУКТОР =========================
VideoStabilizer::VideoStabilizer() 
    : rawFramesQueue(MAX_QUEUE_SIZE),
      processedFramesQueue(MAX_QUEUE_SIZE),
      running(false),
      tauStab(TAU_STAB_MAX / 4),
      kSwitch(0.1),
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
        maxCornersConfig, qualityLevelConfig, minDistanceConfig,
        blockSizeConfig, useHarrisDetectorConfig, harrisKConfig
    );

    cout << "Detector created with maxCorners=" << maxCornersConfig << endl;
    cout << "Memory limits: Queue size=" << MAX_QUEUE_SIZE 
         << ", Frame buffer=" << MAX_FRAME_BUFFER_SIZE << endl;
}

VideoStabilizer::~VideoStabilizer() {
    stop();
}

void VideoStabilizer::setUseFP16(bool value) {
    useFP16Warp = value;
    if (value) {
        cout << "[GPU] FP16 оптимизация активирована для warpAffine()" << endl;
    }
}

void VideoStabilizer::start(bool useVideo, const string& imageFolderPath) {
    running = true;
    
    workers.emplace_back(&VideoStabilizer::captureThreadWrapper, this, useVideo, imageFolderPath);
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

void VideoStabilizer::stop() {
    running = false;
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

void VideoStabilizer::captureThreadWrapper(bool useVideo, const string& imageFolderPath) {
    if (captureCore >= 0) setCurrentThreadAffinity(captureCore);
    captureThread(useVideo, imageFolderPath);
}

void VideoStabilizer::detectionAndTrackingThreadWrapper() {
    if (detectionCore >= 0) setCurrentThreadAffinity(detectionCore);
    detectionAndTrackingThread();
}

void VideoStabilizer::stabilizationThreadWrapper() {
    if (stabilizationCore >= 0) setCurrentThreadAffinity(stabilizationCore);
    stabilizationThread();
}

void VideoStabilizer::displayThreadWrapper() {
    if (displayCore >= 0) setCurrentThreadAffinity(displayCore);
    displayThread();
}

// ========================= ПОТОК ЗАХВАТА КАДРОВ =========================
void VideoStabilizer::captureThread(bool useVideo, const string& imageFolderPath) {
    cv::ocl::setUseOpenCL(false);
    cout << "[CAP] === Capture thread started ===" << endl;
    cout << "[CAP] Mode: " << (useVideo ? "Camera/Video" : "Image sequence") << endl;
    
    VideoCapture cap;
    double videoFPS = 15.0;
    
    if (useVideo) {
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
        Mat firstFrameCPU;
        loadImage(firstFrameCPU, 0, imageFolderPath);
        if (firstFrameCPU.empty()) {
            cerr << "Cannot find any images in the sequence" << endl;
            running = false;
            return;
        }
        frameSize = Size(firstFrameCPU.cols, firstFrameCPU.rows);
        videoFPS = 15.0;
    }
    
    if (cap.isOpened() && !useVideo) {
        double fpsFromFile = cap.get(CAP_PROP_FPS);
        if (fpsFromFile > 0) videoFPS = fpsFromFile;
    }
    
    bool shouldThrottle = !useVideo;
    double frameDelayMs = 1000.0 / videoFPS;
    
    frameWidth = frameSize.width;
    frameHeight = frameSize.height;
    
    a = frameSize.width;
    b = frameSize.height;
    
    // Используем глобальную framePart — член класса ещё не инициализирован в потоке
    double fp = ::framePart;
    
    roi.x = static_cast<int>(frameWidth * ((1.0 - fp) / 2.0));
    roi.y = static_cast<int>(frameHeight * ((1.0 - fp) / 2.0));
    roi.width = static_cast<int>(frameWidth * fp);
    roi.height = static_cast<int>(frameHeight * fp);
    
    cout << "[CAP] ROI: x=" << roi.x << " y=" << roi.y << " w=" << roi.width << " h=" << roi.height << endl;
    
    roi.x = max(0, min(roi.x, frameWidth - roi.width));
    roi.y = max(0, min(roi.y, frameHeight - roi.height));
    roi.width = min(roi.width, frameWidth - roi.x);
    roi.height = min(roi.height, frameHeight - roi.y);
    
    cout << "Resolution: " << frameWidth << "x" << frameHeight << endl;
    cout << "ROI: x=" << roi.x << " y=" << roi.y << " w=" << roi.width << " h=" << roi.height << endl;
    
    int frameId = 0;
    auto lastFpsTime = chrono::steady_clock::now();
    auto lastFrameTime = chrono::steady_clock::now();
    int frameCount = 0;
    int consecutiveSkips = 0;
    
    while (running) {
        auto startTimeCap = chrono::steady_clock::now();
        
        if (shouldThrottle && frameId > 0) {
            auto now = chrono::steady_clock::now();
            auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFrameTime).count();
            if (elapsed < frameDelayMs) {
                int sleepMs = static_cast<int>(frameDelayMs - elapsed);
                if (sleepMs > 0 && sleepMs < 100) {
                    this_thread::sleep_for(chrono::milliseconds(sleepMs));
                }
            }
        }
        
        int totalLag = rawFramesQueue.size() + processedFramesQueue.size();
        int maxAllowedLag = shouldThrottle ? MAX_PROCESSING_LAG * 2 : MAX_PROCESSING_LAG;
        
        if (totalLag > maxAllowedLag) {
            consecutiveSkips++;
            if (consecutiveSkips > SKIP_FRAMES_THRESHOLD) {
                framesSkipped++;
                if (useVideo) {
                    Mat dummy;
                    cap.read(dummy);
                }
                consecutiveSkips = 0;
            }
            this_thread::sleep_for(chrono::milliseconds(5));
            continue;
        }
        
        consecutiveSkips = 0;
        
        FrameData frameData;
        frameData.frameId = frameId++;
        
        if (useVideo) {
            bool frameRead = cap.read(frameData.frameCPU);
            if (!frameRead || frameData.frameCPU.empty()) {
                cerr << "Failed to read frame from camera, reopening..." << endl;
                cap.release();
                cap.open(videoSource);
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
        } else {
            if (cap.isOpened()) {
                bool frameRead = cap.read(frameData.frameCPU);
                if (!frameRead || frameData.frameCPU.empty()) {
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
                loadImage(frameData.frameCPU, frameData.frameId % 1200, imageFolderPath);
                if (frameData.frameCPU.empty()) {
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
        
        Mat compressed, gray;
        Size compressedSize(frameWidth / compressionConfig, frameHeight / compressionConfig);
        if (compressedSize.width <= 0) compressedSize.width = 1;
        if (compressedSize.height <= 0) compressedSize.height = 1;
        
        resize(frameData.frameCPU, compressed, compressedSize, 0, 0, INTER_AREA);
        cvtColor(compressed, gray, COLOR_BGR2GRAY);
        gray.copyTo(frameData.grayCPU);
        
        bool pushed = false;
        int retryCount = 0;
        const int MAX_RETRIES = 3;
        
        while (!pushed && retryCount < MAX_RETRIES && running) {
            if (rawFramesQueue.push(move(frameData))) {
                pushed = true;
            } else {
                retryCount++;
                this_thread::sleep_for(chrono::milliseconds(5));
                if (retryCount < MAX_RETRIES && !pushed) {
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
            framesSkipped++;
            if (framesSkipped % 30 == 0) {
                cout << "[CAP] Warning: Dropped frame " << frameData.frameId 
                     << " (queue full). Total skipped: " << framesSkipped << endl;
            }
        } else {
            cout << "[CAP] Frame " << frameData.frameId << " captured and pushed to rawFramesQueue (size: " 
                 << rawFramesQueue.size() << ")" << endl;
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
        lastFrameTime = chrono::steady_clock::now();
    }
    
    if (cap.isOpened()) cap.release();
    cout << "Capture thread stopped. Total frames captured: " << frameCount 
         << ", Skipped: " << framesSkipped << endl;
}

// ========================= ПОТОК ДЕТЕКТИРОВАНИЯ И ТРЕКИНГА =========================
void VideoStabilizer::detectionAndTrackingThread() {
    cv::ocl::setUseOpenCL(false);
    cout << "[DETECT] === Detection and Tracking thread started (CPU only) ===" << endl;
    
    TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
    Size winSize(winSizeConfig, winSizeConfig);
    int consecutiveFailures = 0;
    const int MAX_CONSECUTIVE_FAILURES = 10;
    int framesProcessedInThread = 0;
    int framesSinceLastRedetection = 0;
    const int REDETECTION_INTERVAL = 15;
    int lastLoggedFrame = -1;
    
    while (running) {
        FrameData frameData;
        
        for (int attempt = 0; attempt < 500 && running; attempt++) {
            if (rawFramesQueue.try_pop(frameData)) break;
            this_thread::sleep_for(chrono::milliseconds(10));
        }
        
        if (!running) break;
        if (frameData.frameCPU.empty()) continue;
        
        auto startTimeDetTrack = chrono::steady_clock::now();
        framesProcessedInThread++;
        
        // ЭТАП 1: Оптический поток
        if (!firstFrameForTracking && !prevPoints.empty() && !frameData.grayCPU.empty() && !prevGrayCPU.empty()) {
            vector<Point2f> nextPoints;
            vector<uchar> status;
            vector<float> err;
            
            try {
                calcOpticalFlowPyrLK(prevGrayCPU, frameData.grayCPU,
                    prevPoints, nextPoints, status, err,
                    winSize, maxLevelConfig, termcrit, 0, 0.001);
            } catch (const exception& e) {
                cerr << "Error in optical flow: " << e.what() << endl;
                frameData.grayCPU.copyTo(prevGrayCPU);
                frameData.points.clear();
                prevPoints.clear();
                trackedPoints.store(0);
                
                while (processedFramesQueue.size() >= MAX_QUEUE_SIZE && running)
                    this_thread::sleep_for(chrono::milliseconds(1));
                processedFramesQueue.push(move(frameData));
                continue;
            }
            
            vector<Point2f> goodNew, goodOld;
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
        
        // ЭТАП 2: Добавление новых точек
        int currentPointCount = frameData.points.size();
        bool needMorePoints = currentPointCount < maxCornersConfig * 0.7;
        bool timeToRedetect = framesSinceLastRedetection >= REDETECTION_INTERVAL && !firstFrameForTracking;
        
        if (firstFrameForTracking || prevPoints.empty() || (needMorePoints && timeToRedetect)) {
            framesSinceLastRedetection = 0;
            
            Mat maskCPU = Mat::zeros(frameData.grayCPU.size(), CV_8U);
            int marginX = frameData.grayCPU.cols / 8;
            int marginY = frameData.grayCPU.rows / 8;
            rectangle(maskCPU, Rect(marginX, marginY,
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
            
            Mat exclusionMask = Mat::zeros(frameData.grayCPU.size(), CV_8U);
            if (!frameData.points.empty()) {
                for (const auto& pt : frameData.points) {
                    circle(exclusionMask, pt, static_cast<int>(minDistanceConfig), Scalar(255), FILLED);
                }
            }
            
            int maxNewPoints = max(1, static_cast<int>(maxCornersConfig * 0.15));
            int targetNewPoints = min(maxNewPoints, maxCornersConfig - static_cast<int>(frameData.points.size()));
            
            if (targetNewPoints > 0 && !keypoints.empty()) {
                sort(keypoints.begin(), keypoints.end(),
                     [](const KeyPoint& a, const KeyPoint& b) {
                         return a.response > b.response;
                     });
                
                vector<Point2f> newPoints;
                for (const auto& kp : keypoints) {
                    if (newPoints.size() >= targetNewPoints) break;
                    int x = static_cast<int>(kp.pt.x);
                    int y = static_cast<int>(kp.pt.y);
                    if (x >= 0 && x < exclusionMask.cols && y >= 0 && y < exclusionMask.rows) {
                        if (exclusionMask.at<uchar>(y, x) == 0) {
                            newPoints.push_back(kp.pt);
                            circle(exclusionMask, kp.pt, static_cast<int>(minDistanceConfig), Scalar(255), FILLED);
                        }
                    }
                }
                
                if (!newPoints.empty()) {
                    frameData.points.insert(frameData.points.end(), newPoints.begin(), newPoints.end());
                    removeFramePoints(frameData.points, minDistanceConfig * 0.8);
                    if (frameData.points.size() > maxCornersConfig)
                        frameData.points.resize(maxCornersConfig);
                }
            }
            
            if (firstFrameForTracking) {
                prevPoints = frameData.points;
                frameData.grayCPU.copyTo(prevGrayCPU);
                firstFrameForTracking = false;
                trackedPoints.store(static_cast<int>(prevPoints.size()));
            }
        } else {
            framesSinceLastRedetection++;
        }
        
        if (!firstFrameForTracking) {
            prevPoints = frameData.points;
            frameData.grayCPU.copyTo(prevGrayCPU);
        }
        
        // Адаптивная настройка детектора
        if (frameData.points.size() < maxCornersConfig / 6) {
            qualityLevelConfig *= 0.98;
            harrisKConfig *= 0.98;
            detector->setQualityLevel(qualityLevelConfig);
            detector->setK(harrisKConfig);
        }
        if (frameData.points.size() > maxCornersConfig * 4 / 5) {
            qualityLevelConfig *= 1.02;
            harrisKConfig *= 1.02;
            detector->setQualityLevel(qualityLevelConfig);
            detector->setK(harrisKConfig);
        }

        auto endTimeDetTrack = chrono::steady_clock::now();
        auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
        processingTimeDetectionTracking = (processingTimeDetectionTracking * 99.0 + durationDetTrack.count() / 1000.0) / 100.0;
        
        while (processedFramesQueue.size() >= MAX_QUEUE_SIZE && running)
            this_thread::sleep_for(chrono::milliseconds(5));
        
        processedFramesQueue.push(move(frameData));
        
        // Логирование каждые 10 кадров
        if (framesProcessedInThread % 10 == 0 && framesProcessedInThread != lastLoggedFrame) {
            cout << "[DETECT] Frame " << frameData.frameId 
                 << " processed | Points tracked: " << frameData.points.size()
                 << " | Queue size: " << processedFramesQueue.size()
                 << " | Total processed: " << framesProcessedInThread << endl;
            lastLoggedFrame = frameData.frameId;
        }
    }
    
    cout << "[DETECT] === Detection and Tracking thread stopped. Processed " << framesProcessedInThread << " frames. ===" << endl;
}

// ========================= ПОТОК СТАБИЛИЗАЦИИ =========================
void VideoStabilizer::stabilizationThread() {
    cv::ocl::setUseOpenCL(USE_OPENCL);
    cout << "[STAB] === Stabilization thread started (GPU enabled) ===" << endl;
    
    int framesStabilized = 0;
    int framesSkippedWiener = 0;
    
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
                
                iirAdaptive(frameData.transformFirstDerivative, oldTransform,
                            frameData.transformSKO, frameData.stabMatrix, tauStab, roi, a, b, kSwitch);
                frameData.transform = oldTransform;
                
                // Рисуем точки на CPU
                if (!frameData.points.empty() && ::framePart > 0.7) {
                    int pointsToShow = min(300, static_cast<int>(frameData.points.size()));
                    for (int i = 0; i < pointsToShow; i++) {
                        Point2f pt = frameData.points[i];
                        Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                      static_cast<int>(pt.y * compressionConfig));
                        circle(frameData.frameCPU, scaledPt, 6, colorRED, -1);
                    }
                }
                
                // Копируем кадр на GPU
                frameData.frameCPU.copyTo(frameData.frameGPU);
                
                LEN = sqrt(frameData.transformFirstDerivative.dx * frameData.transformFirstDerivative.dx + 
                    frameData.transformFirstDerivative.dy * frameData.transformFirstDerivative.dy) * D;

                if (frameData.transformFirstDerivative.dx == 0.0)
                    THETA = (frameData.transformFirstDerivative.dy > 0.0) ? 90.0 : -90.0;
                else
                    THETA = atan(frameData.transformFirstDerivative.dy / frameData.transformFirstDerivative.dx) * RAD_TO_DEG;
                
                // ==== Винеровская фильтрация (оригинальное условие: framePart < 0.65 && wiener && !wiener — всегда false) ====
                if (::framePart < 0.65 && wiener && !wiener)
                {
                    UMat zeroMatH(cv::Size(frameWidth, frameHeight), CV_32F, Scalar(0)), complexH;
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
                
                // Стабилизация (всегда выполняется)
                UMat stabilizedFrame, croppedFrame;
                warpAffineOptimized(frameData.frameGPU, stabilizedFrame, frameData.stabMatrix, frameSize, 
                                   cv::INTER_LINEAR, cv::BORDER_CONSTANT, Scalar(), useFP16Warp);
                croppedFrame = stabilizedFrame(roi);
                
                // Копируем результат обратно на CPU для отображения
                croppedFrame.copyTo(frameData.frameCPU);
                // ====================================
                
                cout << "[STAB] Frame " << frameData.frameId 
                     << " stabilized | Points: " << frameData.points.size()
                     << " | Transform: dx=" << frameData.transform.dx 
                     << " dy=" << frameData.transform.dy 
                     << " da=" << frameData.transform.da * RAD_TO_DEG << "deg"
                     << " | Queue size: " << processedFramesQueue.size() << endl;
                
            } catch (const exception& e) {
                cerr << "[STAB] ERROR in stabilization: " << e.what() << endl;
                // Даже при ошибке добавляем кадр в буфер, чтобы не блокировать display
            }
        }
        
        // Добавляем кадр в буфер для отображения (всегда, даже при ошибках)
        {
            lock_guard<mutex> lock(bufferMutex);
            
            if (frameBuffer.size() >= MAX_FRAME_BUFFER_SIZE) {
                int oldestFrameId = frameBuffer.begin()->first;
                frameBuffer.erase(oldestFrameId);
            }
            
            frameBuffer[frameData.frameId] = move(frameData);
            frameReadyCV.notify_one();
            
            cout << "[STAB] Frame " << frameData.frameId 
                 << " pushed to display buffer (size: " << frameBuffer.size() 
                 << ", nextDisplayId: " << nextDisplayFrameId.load() << ")" << endl;
            
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
    
    cout << "[STAB] === Stabilization thread stopped. Stabilized " << framesStabilized 
         << " frames, Wiener failures: " << framesSkippedWiener << " ===" << endl;
}

// ========================= ПОТОК ОТОБРАЖЕНИЯ =========================
void VideoStabilizer::displayThread() {
    cv::ocl::setUseOpenCL(false);
    cout << "[DISPLAY] === Display thread started ===" << endl;
    
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
    int lastLoggedDisplay = -1;
    
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
                cout << "[DISPLAY] Frame " << frameData.frameId << " received from buffer (nextDisplayId now: " 
                     << nextDisplayFrameId.load() << ")" << endl;
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
                        cout << "[DISPLAY] Frame " << frameData.frameId << " (skipped " 
                             << (nextAvailable - frameData.frameId) << " frames) from buffer" << endl;
                    } else if (!frameBuffer.empty()) {
                        cout << "[DISPLAY] WARNING: Large gap detected, resetting nextDisplayId to " 
                             << frameBuffer.begin()->first << " (buffer size: " << frameBuffer.size() << ")" << endl;
                        nextDisplayFrameId = frameBuffer.begin()->first;
                        continue;
                    }
                }
                
                if (!gotFrame) {
                    // Периодически логируем состояние очереди
                    if (displayedFrames % 50 == 0) {
                        cout << "[DISPLAY] No frame available, buffer size: " << frameBuffer.size() 
                             << ", nextDisplayId: " << nextDisplayFrameId.load() << endl;
                    }
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
        
        Mat displayFrame;
        resize(frameData.frameCPU, displayFrame, Size(frameWidth, frameHeight), INTER_AREA);
        
        string infoText = format("FPS: %d | nsr: %1.2f | D: %1.2f | part: %1.2f | Skip: %d",
                                fps.load(), nsr, D, ::framePart, framesSkipped.load());

string infoLatencies = format("Cap: %2.1f | D+T: %2.1f | Stab: %2.1f | Q: %ld / %ld ",
                                    processingTimeCapture, processingTimeDetectionTracking, 
                                    processingTimeStabilization,
                                    (long)rawFramesQueue.size(), (long)processedFramesQueue.size());
        

        // отладочная инфомация

        cout << "Frame ID: " << frameData.frameId << " | Tracked points: " << trackedPoints.load() 
             << " | Transform: dx=" << frameData.transform.dx 
             << ", dy=" << frameData.transform.dy 
             << ", da=" << frameData.transform.da * RAD_TO_DEG 
             << " | First Derivative: dx=" << frameData.transformFirstDerivative.dx 
             << ", dy=" << frameData.transformFirstDerivative.dy 
             << ", da=" << frameData.transformFirstDerivative.da * RAD_TO_DEG 
             << endl;
             
        putText(displayFrame, infoText, Point(10, 50 * frameWidth / 800),
               FONT_HERSHEY_SIMPLEX, 0.5 * frameWidth / 800, colorBLUE, 2 * frameWidth / 800);
        putText(displayFrame, infoLatencies, Point(10, 100 * frameWidth / 800),
               FONT_HERSHEY_SIMPLEX, 0.5 * frameWidth / 800, colorBLUE, 2 * frameWidth / 800);
        
        imshow(windowName, displayFrame);

        
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
            if (::framePart < 0.95) {
                ::framePart *= 1.01;
                if (::framePart > 0.9) ::framePart = 0.9;
                roi.x = frameWidth * ((1.0 - ::framePart) / 2.0);
                roi.y = frameHeight * ((1.0 - ::framePart) / 2.0);
                roi.width = frameWidth * ::framePart;
                roi.height = frameHeight * ::framePart;
            }
        } else if (key == 'w' || key == 'W') {
            if (::framePart > 0.2) {
                ::framePart *= 0.99;
                if (::framePart < 0.05) ::framePart = 0.05;
                roi.x = frameWidth * ((1.0 - ::framePart) / 2.0);
                roi.y = frameHeight * ((1.0 - ::framePart) / 2.0);
                roi.width = frameWidth * ::framePart;
                roi.height = frameHeight * ::framePart;
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
    cout << "[DISPLAY] === Display thread stopped. Displayed " << displayedFrames << " frames. ===" << endl;
}

// ========================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =========================
void VideoStabilizer::loadImage(Mat& image, int frame_id, const string& filepath) {
    ::loadImage(image, frame_id, filepath);
}

void VideoStabilizer::removeFramePoints(vector<Point2f>& points, double minDistance) {
    ::removeFramePoints(points, minDistance);
}

// ========================= IIR АДАПТИВНЫЙ ФИЛЬТР =========================
void iirAdaptive(TransformParam& transformsFirstDerivative, TransformParam& transforms,
                 TransformParam& transformSKO, UMat& stabMatrix,
                 double& tauStab, Rect& roi, int a, int b, double& kSwitch) {
    if (abs(transformsFirstDerivative.dx) < 4.0 * transformSKO.dx + a / 8) {
        transforms.dx = kSwitch * (transforms.dx * (tauStab - 1.0) / tauStab + kSwitch * transformsFirstDerivative.dx);
    }

    if (abs(transformsFirstDerivative.dy) < 4.0 * transformSKO.dy + b / 8) {
        transforms.dy = kSwitch * (transforms.dy * (tauStab - 1.0) / tauStab + kSwitch * transformsFirstDerivative.dy);
    }

    if (abs(transformsFirstDerivative.da) < 4.0 * transformSKO.da + 0.1) {
        transforms.da = kSwitch * (transforms.da * (tauStab - 1.0) / tauStab + kSwitch * transformsFirstDerivative.da);
    }

    if (transforms.da > CV_PI) transforms.da -= CV_PI;
    if (transforms.da < -CV_PI) transforms.da += CV_PI;

    if (tauStab < TAU_STAB_MAX / 4) tauStab *= 1.2;
    if (tauStab < TAU_STAB_MAX / 2 && !(abs(transforms.dx) > a / 2 || abs(transforms.dy) > b / 2)) tauStab *= 1.1;
    if (tauStab < TAU_STAB_MAX && !(abs(transforms.dx) > a / 3 || abs(transforms.dy) > b / 3)) {
        tauStab *= 1.1;
        if (tauStab > TAU_STAB_MAX) tauStab = TAU_STAB_MAX;
    }

    if (roi.x + (int)transforms.dx < 0) {
        transforms.dx = double(1 - roi.x);
        if (tauStab > TAU_STAB_MAX / 2) {
            tauStab *= 0.9;
            kSwitch *= 0.95;
        }
    } else if (roi.x + roi.width + (int)transforms.dx >= a) {
        transforms.dx = double(a - roi.x - roi.width);
        if (tauStab > TAU_STAB_MAX / 2) {
            tauStab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (roi.y + (int)transforms.dy < 0) {
        transforms.dy = double(1 - roi.y);
        if (tauStab > TAU_STAB_MAX / 8) {
            tauStab *= 0.9;
            kSwitch *= 0.95;
        }
    } else if (roi.y + roi.height + (int)transforms.dy >= b) {
        transforms.dy = double(b - roi.y - roi.height);
        if (tauStab > TAU_STAB_MAX / 2) {
            tauStab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (kSwitch < 1.0) tauStab *= (4.0 + kSwitch) / 5.0;

    transformSKO.dx = (1.0 - 0.1) * transformSKO.dx + 0.1 * abs(transformsFirstDerivative.dx);
    transformSKO.dy = (1.0 - 0.1) * transformSKO.dy + 0.1 * abs(transformsFirstDerivative.dy);
    transformSKO.da = (1.0 - 0.1) * transformSKO.da + 0.1 * abs(transformsFirstDerivative.da);

    transforms.getTransformInvert(stabMatrix);
}
