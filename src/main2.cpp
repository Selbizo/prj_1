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

using namespace cv;
using namespace std;
namespace fs = filesystem;

// ========================= КОНСТАНТЫ И КОНФИГУРАЦИЯ =========================
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
// const bool multiScreen = true;
const bool recordEnable = false;
const int compressionConfig = 3; // Сжатие для обработки
//const int outputResolution = 720; // Разрешение вывода

// Настройки детектора
int maxCornersConfig = 200 / compressionConfig;
double qualityLevelConfig = 0.005 / compressionConfig;
const double minDistanceConfig = 4.0;
const int blockSizeConfig = 9;
const bool useHarrisDetectorConfig = true;
double harrisKConfig = qualityLevelConfig;

// Настройки оптического потока
const int winSizeConfig = blockSizeConfig;
const int maxLevelConfig = 5;
const int itersConfig = 6;

// Источник видео (измените на свой)
const string videoSource = "http://192.168.0.102:4747/video";

// Убрали фиксированный путь к файлам
string filepath = "";

// ========================= СТРУКТУРЫ ДАННЫХ =========================

struct TransformParam {
    double dx;
    double dy;
    double da; // угол в радианах
    
    TransformParam() : dx(0), dy(0), da(0) {}
    TransformParam(double _dx, double _dy, double _da) 
        : dx(_dx), dy(_dy), da(_da) {}
    
    void getTransform(Mat& T) const {
        T = Mat::zeros(2, 3, CV_64F); // ИНИЦИАЛИЗИРУЕМ МАТРИЦУ
        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = -sin(da);
        T.at<double>(0, 2) = dx;
        T.at<double>(1, 0) = sin(da);
        T.at<double>(1, 1) = cos(da);
        T.at<double>(1, 2) = dy;
    }
    
    void getTransformInvert(Mat& T) const {
        T = Mat::zeros(2, 3, CV_64F); // ИНИЦИАЛИЗИРУЕМ МАТРИЦУ
        T.at<double>(0, 0) = cos(da);
        T.at<double>(0, 1) = sin(da);
        T.at<double>(0, 2) = -dx;
        T.at<double>(1, 0) = -sin(da);
        T.at<double>(1, 1) = cos(da);
        T.at<double>(1, 2) = -dy;
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

// ========================= ОСНОВНЫЕ ФУНКЦИИ =========================

class VideoStabilizer {
private:
    // Очереди конвейерной обработки
    ThreadSafeQueue rawFramesQueue;          // Захват -> Объединенный поток
    ThreadSafeQueue processedFramesQueue;    // Объединенный поток -> Стабилизация
    
    // Буфер для сохранения порядка кадров при отображении
    unordered_map<int, FrameData> frameBuffer;
    mutex bufferMutex;
    atomic<int> nextDisplayFrameId{0};
    condition_variable frameReadyCV;
    
    // Общие ресурсы
    mutex resourcesMutex;
    Ptr<GFTTDetector> detector;
    
    // Для отслеживания точек между кадрами
    Mat prevGray;
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
    
    // Для отладки
    atomic<bool> debugMode;
    
    // Оптимизация производительности
    atomic<int> processingLag{0};
    const int MAX_PROCESSING_LAG = 8;
    const int MAX_BUFFER_SIZE = 10;
    
    double processingTimeCapture;
    double processingTimeDetectionTracking;
    double processingTimeStabilization;
    double processingTimeImshow;

public:
    // Потоки обработки
    vector<thread> workers;
    atomic<bool> running;

    VideoStabilizer() 
        : running(false), 
          tauStab(100.0), 
          kSwitch(0.1), 
          framePart(0.7),
          trackedPoints(0),
          framesProcessed(0),
          debugMode(false) {
        
        //Инициализация детектора
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
        
        // Запуск потоков обработки (объединенный поток заменяет два старых)
        workers.emplace_back(&VideoStabilizer::captureThread, this, useCamera, imageFolderPath);
        workers.emplace_back(&VideoStabilizer::detectionAndTrackingThread, this); // Объединенный поток
        workers.emplace_back(&VideoStabilizer::stabilizationThread, this);
        workers.emplace_back(&VideoStabilizer::displayThread, this);
        
        cout << "Video stabilizer started with " << workers.size() << " threads" << endl;
    }
    
    void stop() {
        running = false;
        
        // Будим все ожидающие потоки
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
        cout << "Video stabilizer stopped. Total frames processed: " << framesProcessed << endl;
    }
    
private:
    // ========================= ПОТОК ЗАХВАТА КАДРОВ =========================
    void captureThread(bool useCamera, const string& imageFolderPath) {
        // Если не используем камеру, читаем изображения из файлов
        if (!useCamera) {
            if (imageFolderPath.empty()) {
                cerr << "File path not specified for image sequence loading" << endl;
                running = false;
                return;
            }
            
            // Проверяем, существует ли директория
            struct stat info;
            if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
                cerr << "Cannot access directory: " << imageFolderPath << endl;
                running = false;
                return;
            }
            
            cout << "Loading image sequence from: " << imageFolderPath << endl;
        }
        
        // Для режима камеры оставляем VideoCapture
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
            
            // Получаем параметры видео с камеры
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
            // Для режима изображений загружаем первое изображение для определения размеров
            Mat firstFrame;
            loadImage(firstFrame, 0, imageFolderPath);
            
            if (firstFrame.empty()) {
                cerr << "Failed to load first image, checking other indices..." << endl;
                // Пробуем найти первое доступное изображение
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
        
        // Инициализация ROI
        roi.x = static_cast<int>(a * ((1.0 - framePart) / 2.0));
        roi.y = static_cast<int>(b * ((1.0 - framePart) / 2.0));
        roi.width = static_cast<int>(a * framePart);
        roi.height = static_cast<int>(b * framePart);
        
        // Убедимся, что ROI находится в границах кадра
        roi.x = max(0, min(roi.x, a - roi.width));
        roi.y = max(0, min(roi.y, b - roi.height));
        roi.width = min(roi.width, a - roi.x);
        roi.height = min(roi.height, b - roi.y);
        
        cout << "Resolution: " << a << "x" << b << endl;
        cout << "ROI: x=" << roi.x << " y=" << roi.y << " w=" << roi.width << " h=" << roi.height << endl;
        
        int frameId = 0;
        auto lastFpsTime = chrono::steady_clock::now();
        int frameCount = 0;
        
        while (running) {
            // Ограничиваем размер очереди, если обработка отстает
            auto startTimeCap = chrono::steady_clock::now();
            int totalLag = rawFramesQueue.size() + processedFramesQueue.size();
            
            if (totalLag > MAX_PROCESSING_LAG) {
                this_thread::sleep_for(chrono::milliseconds(5));
                cout << "MAX_PROCESSING_LAG" << endl;
                continue;
            }
            
            FrameData frameData;
            frameData.frameId = frameId++;
            
            // Захват кадра в зависимости от режима
            Mat frame;
            bool frameRead = false;
            
            if (useCamera) {
                // Режим камеры
                frameRead = cap.read(frame);
                if (!frameRead) {
                    cerr << "Failed to read frame from camera" << endl;
                    continue;
                }
            } else {
                // Режим чтения изображений из файлов
                loadImage(frameData.frame, frameData.frameId%1200, imageFolderPath);
                
                if (frameData.frame.empty()) {
                    // Если изображение не найдено, пробуем следующий индекс
                    cerr << "Failed to load image for frame " << frameData.frameId 
                         << ", trying next..." << endl;
                    loadImage(frameData.frame, frameData.frameId%1200 + 1, imageFolderPath);
                    if (frameData.frame.empty()) {
                        // Если следующее тоже не найдено, возможно, последовательность закончилась
                        cout << "Image sequence ended or no more images available" << endl;
                        running = false; // Останавливаем поток
                        break;
                    }
                    frameData.frameId++; // Увеличиваем ID, если загрузили следующий кадр
                }
                
                // Конвертируем Mat в Mat для дальнейшей обработки
                frameData.frame.copyTo(frame);
                frameRead = !frame.empty();
            }
            
            if (!frameRead || frame.empty()) {
                cerr << "Empty frame captured" << endl;
                continue;
            }
            
            // Если размер изменился, обновляем
            if (frame.cols != a || frame.rows != b) {
                cout << "Frame size changed from " << a << "x" << b 
                     << " to " << frame.cols << "x" << frame.rows << endl;
                a = frame.cols;
                b = frame.rows;
                frameSize = Size(a, b);
                
                // Обновляем ROI
                roi.x = static_cast<int>(a * ((1.0 - framePart) / 2.0));
                roi.y = static_cast<int>(b * ((1.0 - framePart) / 2.0));
                roi.width = static_cast<int>(a * framePart);
                roi.height = static_cast<int>(b * framePart);
                
                roi.x = max(0, min(roi.x, a - roi.width));
                roi.y = max(0, min(roi.y, b - roi.height));
                roi.width = min(roi.width, a - roi.x);
                roi.height = min(roi.height, b - roi.y);
            }
            
            // Для режима камеры копируем в frameData.frame
            if (useCamera) {
                frame.copyTo(frameData.frame);
            }
            
            // Создаем уменьшенную серую версию для обработки
            Mat compressed, gray;
            Size compressedSize(a / compressionConfig, b / compressionConfig);
            if (compressedSize.width <= 0) compressedSize.width = 1;
            if (compressedSize.height <= 0) compressedSize.height = 1;
            
            resize(frameData.frame, compressed, compressedSize, 0, 0, INTER_LINEAR);
            cvtColor(compressed, gray, COLOR_BGR2GRAY);
            gray.copyTo(frameData.gray);
            
            // Добавляем в очередь для обработки
            rawFramesQueue.push(move(frameData));
            
            // Расчет FPS
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
            VideoStabilizer::processingTimeCapture = (VideoStabilizer::processingTimeCapture*99.0 + durationCap.count() / 1000.0)/100.0;
        }
        
        if (useCamera) {
            cap.release();
        }
        
        cout << "Capture thread stopped" << endl;
    }
    
    // ========================= ОБЪЕДИНЕННЫЙ ПОТОК ДЕТЕКТИРОВАНИЯ И ОТСЛЕЖИВАНИЯ =========================
    void detectionAndTrackingThread() {
        cout << "Detection and Tracking thread started" << endl;
        
        TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
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
            
            // Часть 1: Детектирование точек (только если нужно)
            if (firstFrameForTracking || prevPoints.empty() || trackedPoints.load() < maxCornersConfig / 5) {
                // Детектирование точек
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
                
                // Конвертация KeyPoint в Point2f
                for (const auto& kp : keypoints) {
                    frameData.points.push_back(kp.pt);
                }
                
                // Ограничение размера пула точек
                if (frameData.points.size() > maxCornersConfig * 4) {
                    frameData.points.erase(frameData.points.begin(), 
                                frameData.points.begin() + (frameData.points.size() - maxCornersConfig));
                }
                
                // Удаление слишком близких точек
                removeFramePoints(frameData.points, minDistanceConfig * 0.8);
                
                if (firstFrameForTracking) {
                    // Первый кадр - просто сохраняем точки
                    frameData.gray.copyTo(prevGray);
                    prevPoints = frameData.points;
                    firstFrameForTracking = false;
                    trackedPoints.store(static_cast<int>(prevPoints.size()));
                    frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                    
                    // Отправляем кадр дальше
                    processedFramesQueue.push(move(frameData));
                    
                    auto endTimeDetTrack = chrono::steady_clock::now();
                    auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
                    VideoStabilizer::processingTimeDetectionTracking = (VideoStabilizer::processingTimeDetectionTracking * 99.0 + durationDetTrack.count() / 1000.0) / 100.0;
                    
                    continue;
                }
            }
            
            // Часть 2: Отслеживание точек (если это не первый кадр)
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
                        termcrit, 0, 0.001
                    );
                } catch (const exception& e) {
                    cerr << "Error in optical flow: " << e.what() << endl;
                    // Сбрасываем состояние
                    frameData.gray.copyTo(prevGray);
                    frameData.points.clear();
                    prevPoints.clear();
                    trackedPoints.store(0);
                    processedFramesQueue.push(move(frameData));
                    continue;
                }
                
                // Фильтрация хороших точек
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
                
                // Оценка аффинного преобразования между кадрами
                if (goodCount >= 6) {
                    Mat T;
                    try {
                        T = estimateAffine2D(goodOld, goodNew, noArray(), RANSAC, 3.0);
                    } catch (const exception& e) {
                        cerr << "Error in estimateAffine2D: " << e.what() << endl;
                        T = Mat();
                    }
                    
                    if (!T.empty() && T.rows == 2 && T.cols == 3) {
                        // Извлечение параметров трансформации
                        double dx = T.at<double>(0, 2) * compressionConfig;
                        double dy = T.at<double>(1, 2) * compressionConfig;
                        double da = atan2(T.at<double>(1, 0), 
                                        T.at<double>(0, 0));
                        
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
                
                // Обновляем данные для следующего кадра
                frameData.gray.copyTo(prevGray);
                prevPoints = goodNew;
                frameData.points = goodNew;
                
                // Обновляем количество отслеживаемых точек
                trackedPoints.store(static_cast<int>(goodNew.size()));
                consecutiveFailures = 0;
            } else {
                // Нет точек для отслеживания
                if (consecutiveFailures++ > MAX_CONSECUTIVE_FAILURES) {
                    // Сбрасываем состояние после серии неудач
                    firstFrameForTracking = true;
                    consecutiveFailures = 0;
                    trackedPoints.store(0);
                    //cout << "Tracking: reset due to consecutive failures" << endl;
                }
                frameData.transformFirstDerivative = TransformParam(0, 0, 0);
            }
            
            // Адаптивная настройка параметров детектора
            if ((frameData.points.size() < maxCornersConfig / 6) && maxCornersConfig > 60) {
                maxCornersConfig *= 0.98;
                maxCornersConfig -= 1;
                if (maxCornersConfig < 60) {
                    maxCornersConfig = 60;
                    qualityLevelConfig *= 0.98;
                    harrisKConfig *= 0.98;
                }
                detector->setMaxFeatures(maxCornersConfig);
                detector->setQualityLevel(qualityLevelConfig);
                detector->setK(harrisKConfig);
            }

            if ((frameData.points.size() > maxCornersConfig * 4 / 5) && (maxCornersConfig < 199)) {
                maxCornersConfig *= 1.05;
                maxCornersConfig += 1;
                if (maxCornersConfig > 200) {
                    maxCornersConfig = 200;
                    qualityLevelConfig *= 1.02;
                    harrisKConfig *= 1.02;
                }
                detector->setMaxFeatures(maxCornersConfig);
                detector->setQualityLevel(qualityLevelConfig);
                detector->setK(harrisKConfig);
            }

            auto endTimeDetTrack = chrono::steady_clock::now();
            auto durationDetTrack = chrono::duration_cast<chrono::microseconds>(endTimeDetTrack - startTimeDetTrack);
            VideoStabilizer::processingTimeDetectionTracking = (VideoStabilizer::processingTimeDetectionTracking * 99.0 + durationDetTrack.count() / 1000.0) / 100.0;
            
            // Отправляем обработанный кадр в следующую очередь конвейера
            processedFramesQueue.push(move(frameData));
        }
        
        cout << "Detection and Tracking thread stopped" << endl;
    }
    
    // ========================= ПОТОК СТАБИЛИЗАЦИИ =========================
    void stabilizationThread() {
        const int HISTORY_SIZE = 10;
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
                    if (VideoStabilizer::kSwitch < 0.01) 
                        VideoStabilizer::kSwitch = 0.01;
                    if (VideoStabilizer::kSwitch < 1.0)
                    {
                        VideoStabilizer::kSwitch *= 1.06;
                        VideoStabilizer::kSwitch += 0.005;
                    }
                    else if (VideoStabilizer::kSwitch > 1.0)
                        VideoStabilizer::kSwitch = 1.0;
                    

                    VideoStabilizer::kSwitch = 1.0;
                    framesProcessed++;
                    
                    // Стабилизация
                    iirAdaptive(frameData.transformFirstDerivative, oldTransform, 
                               frameData.transformSKO, frameData.stabMatrix, VideoStabilizer::tauStab, VideoStabilizer::roi, a, b, VideoStabilizer::kSwitch);
                    
                    frameData.transform = oldTransform;
                    
                    // Применяем стабилизацию
                    Mat stabilizedFrame, croppedFrame;

                    // Рисуем точки
                    if (!frameData.points.empty()) {
                        int pointsToShow = min(20, static_cast<int>(frameData.points.size()));
                        for (int i = 0; i < pointsToShow; i++) {
                            Point2f pt = frameData.points[i];
                            Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                        static_cast<int>(pt.y * compressionConfig));
                            circle(frameData.frame, scaledPt, 2, colorRED, -1);
                        }
                    }

                    warpAffine(frameData.frame, stabilizedFrame, frameData.stabMatrix, frameSize);
                    croppedFrame = stabilizedFrame(roi);
                    croppedFrame.copyTo(frameData.frame);
                    framesStabilized++;
                    
                } catch (const exception& e) {
                    cerr << "Error in stabilization: " << e.what() << endl;
                }
            }

            auto endTimeStab = chrono::steady_clock::now();
            auto durationStab = chrono::duration_cast<chrono::microseconds>(endTimeStab - startTimeStab);
            VideoStabilizer::processingTimeStabilization = (VideoStabilizer::processingTimeStabilization*99.0 + durationStab.count() / 1000.0)/100.0;
            
            // Добавляем кадр в буфер в правильном порядке
            {
                lock_guard<mutex> lock(bufferMutex);
                frameBuffer[frameData.frameId] = move(frameData);
                
                // Уведомляем поток отображения
                frameReadyCV.notify_one();
                
                // Очищаем старые кадры из буфера
                if (frameBuffer.size() > MAX_BUFFER_SIZE * 2) {
                    // Удаляем самые старые кадры
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

        }
        
        cout << "Stabilization thread stopped. Stabilized " << framesStabilized 
             << " frames total." << endl;
    }
    
    // ========================= ПОТОК ОТОБРАЖЕНИЯ =========================
    void displayThread() {
        const string windowName = "Video Stabilization";
        namedWindow(windowName, WINDOW_NORMAL);
        resizeWindow(windowName, 1280, 720);
        
        // Для записи видео (если включено)
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
                
                // Ждем, пока появится следующий кадр в правильном порядке
                auto it = frameBuffer.find(nextDisplayFrameId);
                if (it != frameBuffer.end()) {
                    frameData = move(it->second);
                    frameBuffer.erase(it);
                    nextDisplayFrameId++;
                    gotFrame = true;
                } else {
                    // Проверяем, есть ли следующий кадр в буфере
                    if (!frameBuffer.empty()) {
                        // Находим следующий доступный кадр
                        int nextAvailable = -1;
                        for (auto& pair : frameBuffer) {
                            if (pair.first >= nextDisplayFrameId) {
                                if (nextAvailable == -1 || pair.first < nextAvailable) {
                                    nextAvailable = pair.first;
                                }
                            }
                        }
                        
                        // Если нашли близкий кадр, перескакиваем на него
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
                    
                    // Если не нашли кадр, ждем
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
            
            // Создаем информационный overlay
            Mat displayFrame;
            resize(frameData.frame, displayFrame, Size(a,b));

            auto endTimeDisp = chrono::steady_clock::now();
            auto durationDisp = chrono::duration_cast<chrono::microseconds>(endTimeDisp - startTimeDisp);
            VideoStabilizer::processingTimeImshow = (VideoStabilizer::processingTimeImshow*99.0 + durationDisp.count() / 1000.0)/100.0;


            string infoText = format("Frame: %d | FPS: %d | Process: %2.1f ms | tauStab: %2.1f, | framePart: %1.2f",
                                    frameData.frameId, fps.load(),
                                    VideoStabilizer::processingTimeCapture + 
                                    VideoStabilizer::processingTimeDetectionTracking + 
                                    VideoStabilizer::processingTimeStabilization + 
                                    VideoStabilizer::processingTimeImshow, 
                                    VideoStabilizer::tauStab, VideoStabilizer::framePart);

            string infoLatencies = format("Capture: %2.1f | Det+Track: %2.1f | Stabilization: %2.1f | Imshow: %2.1f",
                                    VideoStabilizer::processingTimeCapture, VideoStabilizer::processingTimeDetectionTracking, 
                                    VideoStabilizer::processingTimeStabilization, VideoStabilizer::processingTimeImshow);

            putText(displayFrame, infoText, Point(10, b-50*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5*a/800, colorGREEN, 2);
            putText(displayFrame, infoLatencies, Point(10, b-100*a/800),
                   FONT_HERSHEY_SIMPLEX, 0.5*a/800, colorGREEN, 2);
            
            // Добавляем информацию о трансформации
            string transformText = format("dX: %.1f dY: %.1f dA: %.1f deg",
                                         frameData.transform.dx, frameData.transform.dy,
                                         frameData.transform.da * RAD_TO_DEG);
            putText(displayFrame, transformText, Point(10, 150),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorYELLOW, 2);
            
            // Отображаем
            imshow(windowName, displayFrame);
            
            // Запись видео
            if (recordEnable && writer.isOpened()) {
                writer.write(displayFrame);
            }
            
            // Расчет FPS отображения
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
            
            // Регулируем задержку для поддержания FPS
            auto frameElapsed = chrono::duration_cast<chrono::milliseconds>(now - lastFrameTime);
            if (frameElapsed.count() < 5) { // ~60 FPS
                this_thread::sleep_for(chrono::milliseconds(5 - frameElapsed.count()));
            }
            lastFrameTime = now;
            
            // Обработка клавиш
            int key = waitKey(1);
            if (key == 27 || key == 'q') { // ESC или Q
                cout << "Exit requested by user" << endl;
                running = false;
                break;
            } else if (key == ' ') { // Пробел - пауза
                cout << "Paused. Press any key to continue..." << endl;
                waitKey(0);
            } else if (key == 'f') { // A - сохранить кадр
                string filename = format("frame_%06d.jpg", frameData.frameId);
                imwrite(filename, displayFrame);
                cout << "Frame saved: " << filename << endl;
            } else if (key == 'd') { // D - переключить режим отладки
                debugMode = !debugMode;
                cout << "Debug mode: " << (debugMode ? "ON" : "OFF") << endl;
            } else if (key == 's' || key == 'S') {
                if (VideoStabilizer::framePart < 0.95)
                {
                    VideoStabilizer::framePart *= 1.01;
                    if (VideoStabilizer::framePart > 0.9)
                        VideoStabilizer::framePart = 0.9;
                    VideoStabilizer::roi.x = a * ((1.0 - VideoStabilizer::framePart) / 2.0);
                    VideoStabilizer::roi.y = b * ((1.0 - VideoStabilizer::framePart) / 2.0);
                    VideoStabilizer::roi.width = a * VideoStabilizer::framePart;
                    VideoStabilizer::roi.height = b * VideoStabilizer::framePart;
                }
            }
            if (key == 'w' || key == 'W')
            {
                if (VideoStabilizer::framePart > 0.2)
                {
                    VideoStabilizer::framePart *= 0.99;
                    if (VideoStabilizer::framePart < 0.05)
                        VideoStabilizer::framePart = 0.05;
                    VideoStabilizer::roi.x = a * ((1.0 - VideoStabilizer::framePart) / 2.0);
                    VideoStabilizer::roi.y = b * ((1.0 - VideoStabilizer::framePart) / 2.0);
                    VideoStabilizer::roi.width = a * VideoStabilizer::framePart;
                    VideoStabilizer::roi.height = b * VideoStabilizer::framePart;
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
    void loadImage(cv::Mat& image_color, int frame_id, std::string filepath)
    {
        char file[200];
        std::string filename;
        
        // Сначала пробуем PNG
        sprintf(file, "image_0/%06d.png", frame_id);
        filename = filepath + std::string(file);
        
        // Проверяем существование файла перед чтением
        std::ifstream file_check(filename.c_str());
        if (file_check.good())
        {
            file_check.close();
            image_color = cv::imread(filename, cv::IMREAD_COLOR);
        }
        
        // Если PNG не найден, пробуем JPG
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

    void removeFramePoints(vector<Point2f>& p0, double minDistance)
    {
        if (p0.empty()) return;

        // Сортировка точек по оси Х
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

    void iirAdaptive(TransformParam& transformsFirtsDerivative, TransformParam& transforms, TransformParam& transformSKO, Mat& stabMatrix, 
        double& tauStab, Rect& roi, const int a, const int b, double& kSwitch)
    {
        if (abs(transformsFirtsDerivative.dx) < 4.0 * transformSKO.dx + a/8)
        {
            transforms.dx = kSwitch * (transforms.dx * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.dx);
        } 

        if (abs(transformsFirtsDerivative.dy) < 4.0 * transformSKO.dy + b/8)
        {
            transforms.dy = kSwitch * (transforms.dy * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.dy);
        }

        if (abs(transformsFirtsDerivative.da) < 4.0 * transformSKO.da + 0.1)
        {
            transforms.da = kSwitch * (transforms.da * (tauStab - 1.0) / tauStab + kSwitch * transformsFirtsDerivative.da);
        }

        if (transforms.da > CV_PI)
            transforms.da -= CV_PI;
        if (transforms.da < -CV_PI)
            transforms.da += CV_PI;

        if (tauStab < 30.0)
            tauStab *= 1.2;

        if (tauStab < 50.0 && !(abs(transforms.dx) > a / 2 || abs(transforms.dy) > b / 2))
            tauStab *= 1.1;

        if (tauStab < 100.0 && !(abs(transforms.dx) > a / 3 || abs(transforms.dy) > b / 3))
        {
            tauStab *= 1.1;
            if (tauStab > 100.0)
                tauStab = 100.0;
        }

        // Проверка границ ROI
        if (roi.x + (int)transforms.dx < 0)
        {
            transforms.dx = double(1 - roi.x);
            if (tauStab > 50) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }
        else if (roi.x + roi.width + (int)transforms.dx >= a)
        {
            transforms.dx = (double)(a - roi.x - roi.width);
            if (tauStab > 50) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (roi.y + (int)transforms.dy < 0)
        {
            transforms.dy = (double)(1 - roi.y);
            if (tauStab > 10) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }
        else if (roi.y + roi.height + (int)transforms.dy >= b)
        {
            transforms.dy = (double)(b - roi.y - roi.height);
            if (tauStab > 50) {
                tauStab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (kSwitch < 1.0)
            tauStab *= (4.0 + kSwitch) / 5.0;

        transformSKO.dx = (1.0 - 0.1) * transformSKO.dx + 0.1 * abs(transformsFirtsDerivative.dx);
        transformSKO.dy = (1.0 - 0.1) * transformSKO.dy + 0.1 * abs(transformsFirtsDerivative.dy);
        transformSKO.da = (1.0 - 0.1) * transformSKO.da + 0.1 * abs(transformsFirtsDerivative.da);

        transforms.getTransformInvert(stabMatrix);
    }
};

// ========================= ОСНОВНАЯ ФУНКЦИЯ =========================

int main() {
    cout << "========================================" << endl;
    cout << "     MULTI-THREADED VIDEO STABILIZER    " << endl;
    cout << "========================================" << endl;
    
    // Запрашиваем у пользователя режим работы
    cout << "Выберите режим работы:" << endl;
    cout << "1. Использовать камеру (по умолчанию)" << endl;
    cout << "2. Читать кадры из папки" << endl;
    cout << "Введите 1 или 2: ";
    
    int choice;
    cin >> choice;
    
    bool useCamera = true;
    string imageFolderPath;
    
    if (choice == 2) {
        useCamera = false;
        
        cout << endl << "Введите путь к папке с кадрами:" << endl;
        cout << "Пример: /home/bananapi/Opencv_projects/dataset/videos/PXL_3/" << endl;
        cout << "Путь: ";
        
        cin.ignore(); // Очищаем буфер ввода
        getline(cin, imageFolderPath);
        
        // Удаляем возможные кавычки в начале и конце
        if (!imageFolderPath.empty()) {
            if (imageFolderPath.front() == '"' || imageFolderPath.front() == '\'') {
                imageFolderPath.erase(0, 1);
            }
            if (imageFolderPath.back() == '"' || imageFolderPath.back() == '\'') {
                imageFolderPath.pop_back();
            }
        }
        
        // Проверяем, есть ли слеш в конце пути
        if (!imageFolderPath.empty() && imageFolderPath.back() != '/') {
            imageFolderPath += '/';
        }
        
        cout << "Путь к кадрам: " << imageFolderPath << endl;
        
        // Проверяем существование директории
        struct stat info;
        if (stat(imageFolderPath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            cerr << "Ошибка: директория не существует или недоступна!\n Использование директории по умолчанию." << endl;
            imageFolderPath = "/home/bananapi/Opencv_projects/dataset/videos/PXL_3/";
        }
    } else {
        cout << "Используется режим камеры" << endl;
    }
    
    // Проверка поддержки OpenCL
    // if (ocl::haveOpenCL()) {
    //     cout << "OpenCL is available" << endl;
    //     ocl::setUseOpenCL(true);
    // } else {
    //     cout << "OpenCL is not available, using CPU" << endl;
    // }
    
    // Создание стабилизатора
    VideoStabilizer stabilizer;
    
    // Запуск системы
    stabilizer.start(useCamera, imageFolderPath);
    
    // Ожидание завершения
    cout << endl << "Управление:" << endl;
    cout << "  ESC или Q - выход" << endl;
    cout << "  Пробел - пауза" << endl;
    cout << "  F - сохранить текущий кадр" << endl;
    cout << "  D - переключить режим отладки" << endl;
    cout << "  S/W - увеличить/уменьшить область кадра" << endl;
    
    // Основной цикл ожидания
    try {
        while (stabilizer.running)
        {
            this_thread::sleep_for(chrono::seconds(10));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    // Остановка системы
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}