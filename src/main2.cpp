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
const bool multiScreen = true;
const bool recordEnable = false;
const int compressionConfig = 1; // Сжатие для обработки
const int outputResolution = 720; // Разрешение вывода

// Настройки детектора
const int maxCornersConfig = 200 / compressionConfig;
const double qualityLevelConfig = 0.005 / compressionConfig;
const double minDistanceConfig = 4.0;
const int blockSizeConfig = 9;
const bool useHarrisDetectorConfig = true;
const double harrisKConfig = qualityLevelConfig;

// Настройки оптического потока
const int winSizeConfig = blockSizeConfig;
const int maxLevelConfig = 5;
const int itersConfig = 6;

// Источник видео (измените на свой)
const string videoSource = "http://192.168.0.102:4747/video";

const string filepath = string("/home/bananapi/Opencv_projects/dataset/videos/PXL_3/");
// const int videoSource = 0; // Для камеры

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
        T.at<double>(0, 0) = cos(-da);
        T.at<double>(0, 1) = -sin(-da);
        T.at<double>(0, 2) = -dx;
        T.at<double>(1, 0) = sin(-da);
        T.at<double>(1, 1) = cos(-da);
        T.at<double>(1, 2) = -dy;
    }
    
    void print() const {
        cout << "TransformPrint: dx=" << dx << " dy=" << dy << " da=" << da * RAD_TO_DEG << " deg" << endl;
    }
    string printString() const {
        //cout << "Transform: dx=" << dx << " dy=" << dy << " da=" << da * RAD_TO_DEG << " deg" << endl;
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

    //Mat stabilizationMatrix;
    int frameId;
    double timestamp;
    
    // FrameData() : frameId(0), timestamp(0), stabilizationMatrix(Mat::zeros(2, 3, CV_64F)) {}
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
    // Потокобезопасные очереди
    ThreadSafeQueue rawFramesQueue;
    ThreadSafeQueue processedFramesQueue;
    ThreadSafeQueue displayQueue;
    

    
    // Общие ресурсы
    mutex resourcesMutex;
    Ptr<FeatureDetector> detector;
    
    // Параметры стабилизации
    double tauStab;
    double kSwitch;
    double framePart;
    Rect roi;
    Size frameSize;
    int a, b; // Ширина и высота кадра
    
    // Статистика
    atomic<int> fps;
    atomic<double> processingTime;
    atomic<int> trackedPoints;
    atomic<int> framesProcessed;
    
    // Для отладки
    atomic<bool> debugMode;
    
public:
    // Потоки обработки
    vector<thread> workers;
    atomic<bool> running;

    VideoStabilizer() 
        : running(false), 
          tauStab(100.0), 
          kSwitch(0.01), 
          framePart(0.8),
          fps(10),
          processingTime(0),
          trackedPoints(0),
          framesProcessed(0),
          debugMode(false) {
        
        // Инициализация детектора
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
    
    void start() {
        running = true;
        
        // Запуск потоков обработки
        workers.emplace_back(&VideoStabilizer::captureThread, this, false);
        workers.emplace_back(&VideoStabilizer::detectionThread, this);
        workers.emplace_back(&VideoStabilizer::trackingThread, this);
        workers.emplace_back(&VideoStabilizer::stabilizationThread, this);
        workers.emplace_back(&VideoStabilizer::displayThread, this);
        
        cout << "Video stabilizer started with " << workers.size() << " threads" << endl;
    }
    
    void stop() {
        running = false;
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
// ========================= ПОТОК ЗАХВАТА КАДРОВ =========================



void captureThread(bool useCamera) {
    // Если не используем камеру, читаем изображения из файлов
    if (!useCamera) {
        if (filepath.empty() || filepath == "0") {
            cerr << "File path not specified for image sequence loading" << endl;
            running = false;
            return;
        }
        
        // Проверяем, существует ли директория
        struct stat info;
        if (stat(filepath.c_str(), &info) != 0 || !(info.st_mode & S_IFDIR)) {
            cerr << "Cannot access directory: " << filepath << endl;
            running = false;
            return;
        }
        
        cout << "Loading image sequence from: " << filepath << endl;
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
        loadImage(firstFrame, 0, filepath);
        
        if (firstFrame.empty()) {
            cerr << "Failed to load first image, checking other indices..." << endl;
            // Пробуем найти первое доступное изображение
            for (int i = 1; i < 100; i++) {
                loadImage(firstFrame, i, filepath);
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
    const int MAX_QUEUE_SIZE = 5;
    
    while (running) {
        // Ограничим размер очереди
        if (rawFramesQueue.size() > MAX_QUEUE_SIZE) {
            this_thread::sleep_for(chrono::milliseconds(50));
            continue;
        }
        if (rawFramesQueue.size() <= MAX_QUEUE_SIZE) {
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
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
        } else {
            // Режим чтения изображений из файлов
            loadImage(frameData.frame, frameData.frameId%3+1, filepath);
            
            if (frameData.frame.empty()) {
                // Если изображение не найдено, пробуем следующий индекс
                cerr << "Failed to load image for frame " << frameData.frameId 
                     << ", trying next..." << endl;
                loadImage(frameData.frame, frameData.frameId + 1, filepath);
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
                cout << "Capture FPS: " << fps << ", Queue size: " 
                     << rawFramesQueue.size() << endl;
            }
        }
    }
        // Ограничение FPS если нужно
        this_thread::sleep_for(chrono::milliseconds(100));
    }
    
    if (useCamera) {
        cap.release();
    }
    
    cout << "Capture thread stopped" << endl;
}
    
    // ========================= ПОТОК ДЕТЕКТИРОВАНИЯ ТОЧЕК =========================
    void detectionThread() {
        
        //int frameSkipCounter = 0;
        //const int DETECTION_SKIP_FRAMES = 1; // Детектировать каждые 5 кадров
        
        cout << "Detection thread started" << endl;
        
        while (running) {
            FrameData frameData;
            if (!rawFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }
            
            auto startTime = chrono::steady_clock::now();
            
            // Проверяем, что кадр не пустой
            if (frameData.gray.empty()) {
                cerr << "Empty gray frame in detection thread" << endl;
                continue;
            }
            // you are here
            // Детектирование точек (не каждый кадр)
            // if (pointPool.size() < maxCornersConfig / 5 ) {
            if (frameData.points.size() < maxCornersConfig / 5 ) {
                Mat mask = Mat::zeros(frameData.gray.size(), CV_8U);
                int marginX = frameData.gray.cols / 4;
                int marginY = frameData.gray.rows / 4;
                rectangle(mask, 
                         Rect(marginX, marginY, 
                              frameData.gray.cols - 2*marginX, 
                              frameData.gray.rows - 2*marginY),
                         Scalar(255), FILLED);
                
                // Детектирование точек
                vector<KeyPoint> keypoints;
                try {
                    detector->detect(frameData.gray, keypoints, mask);
                } catch (const exception& e) {
                    cerr << "Error in detector: " << e.what() << endl;
                    continue;
                }
                
                // Конвертация в Point2f
                //vector<Point2f> newPoints;

                for (const auto& kp : keypoints) {
                    frameData.points.push_back(kp.pt); // нужно добавить в старые точки
                }
                                
                // Ограничение размера пула
                if (frameData.points.size() > maxCornersConfig * 2) {
                    frameData.points.erase(frameData.points.begin(), 
                                   frameData.points.begin() + (frameData.points.size() - maxCornersConfig));
                }
                //you are here
                // Удаление слишком близких точек

                removeFramePoints(frameData.points, minDistanceConfig*0.8);

                if (debugMode && frameData.frameId == 0) {
                    cout << "Detection: found " << keypoints.size() << " keypoints, pool size: " 
                         << frameData.points.size() << endl;
                }
                trackedPoints = static_cast<int>(frameData.points.size());
            }
            
            //frameSkipCounter++;
            
            
            auto endTime = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
            processingTime = duration.count() / 1000.0;
            
            // Отправляем в очередь для отслеживания
            processedFramesQueue.push(move(frameData));
        }
        
        cout << "Detection thread stopped" << endl;
    }
    
    // ========================= ПОТОК ОТСЛЕЖИВАНИЯ ТОЧЕК =========================
    void trackingThread() {
        Mat prevGray;
        vector<Point2f> prevPoints;
        bool firstFrame = true;
        int consecutiveFailures = 0;
        const int MAX_CONSECUTIVE_FAILURES = 10;
        
        TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 20, 0.03);
        Size winSize(winSizeConfig, winSizeConfig);
        
        cout << "Tracking thread started" << endl;
        
        while (running) {
            FrameData frameData;
            if (!processedFramesQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }
            
            auto startTime = chrono::steady_clock::now();
            
            if (firstFrame) {
                // Первый кадр - просто сохраняем
                if (!frameData.gray.empty()) {
                    frameData.gray.copyTo(prevGray);
                    prevPoints = frameData.points;
                    firstFrame = false;
                    cout << "Tracking: first frame initialized with " 
                         << prevPoints.size() << " points" << endl;
                }
                continue;
            }
            
            // Проверяем, что есть предыдущие точки для отслеживания
            if (prevPoints.empty() || frameData.gray.empty() || prevGray.empty()) {
                if (consecutiveFailures++ > MAX_CONSECUTIVE_FAILURES) {
                    // Сбрасываем состояние
                    firstFrame = true;
                    consecutiveFailures = 0;
                    cout << "Tracking: reset due to consecutive failures" << endl;
                }
                continue;
            }
            
            consecutiveFailures = 0;
            
            // Отслеживание точек оптическим потоком
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
            
            if (debugMode && frameData.frameId % 1 == 0) {
                cout << "Tracking: " << goodCount << "/" << status.size() 
                     << " points tracked successfully" << endl;
            }
            
            // Оценка аффинного преобразования
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
                    //frameData.transformFirstDerivative = TransformParam(0.0, 0.0, 0.1);
                    
                    // Сохраняем матрицу стабилизации
                    //T.copyTo(frameData.stabilizationMatrix);
                    
                    if (debugMode && frameData.frameId % 1 == 0) {
                        //frameData.transform.print();
                        frameData.transformFirstDerivative.print();
                    }
                } else {
                    if (debugMode) {
                        cout << "Tracking: transform estimation failed" << endl;
                    }
                    frameData.transformFirstDerivative = TransformParam(0, 0, 0);
                }
            } else {
                if (debugMode && frameData.frameId % 30 == 0) {
                    cout << "Tracking: not enough points for transform (" 
                         << goodCount << " < 6)" << endl;
                }
                frameData.transformFirstDerivative = TransformParam(10, 0, 0);
            }
            
            // Обновляем для следующего кадра
            frameData.gray.copyTo(prevGray);
            prevPoints = frameData.points;
            
            auto endTime = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
            
            // Отправляем в очередь стабилизации
            displayQueue.push(move(frameData));
        }
        
        cout << "Tracking thread stopped" << endl;
    }
    
    // ========================= ПОТОК СТАБИЛИЗАЦИИ =========================
    void stabilizationThread() {
        //vector<TransformParam> transformHistory;
        const int HISTORY_SIZE = 10;
        int framesStabilized = 0;
        
        cout << "Stabilization thread started" << endl;
        
        while (running) {
            FrameData frameData;
            if (!displayQueue.wait_and_pop(frameData)) {
                if (!running) break;
                continue;
            }

            if (!frameData.frame.empty()) {
                try {
                    if (kSwitch < 0.01) 
                        kSwitch = 0.01;
                    if (kSwitch < 1.0)
                    {
                        kSwitch *= 1.06;
                        kSwitch += 0.005;
                    }
                    else if (kSwitch > 1.0) kSwitch = 1.0;
                    framesProcessed++;
                    // auto startTime = chrono::steady_clock::now();
                    iirAdaptive(frameData.transformFirstDerivative, frameData.transform, frameData.transformSKO, 
                    tauStab, roi, a, b, kSwitch);
                    // Вычисляем матрицу стабилизации (инверсия сглаженной трансформации)
                    Mat stabMatrix;
                    frameData.transform.getTransform(stabMatrix);
                    //frameData.transform.print();
                    // Применяем стабилизацию
                    Mat stabilizedFrame;
                    warpAffine(frameData.frame, stabilizedFrame, stabMatrix, frameSize);

                    // Обрезаем по ROI (если ROI валиден)
                    if (true || roi.width > 0 && roi.height > 0 && 
                        roi.x >= 0 && roi.y >= 0 &&
                        roi.x + roi.width <= stabilizedFrame.cols &&
                        roi.y + roi.height <= stabilizedFrame.rows) {
                        
                        //Mat croppedFrame = stabilizedFrame(roi);
                        
                        // Масштабируем обратно к исходному размеру
                        //Mat finalFrame;
                        //resize(croppedFrame, finalFrame, frameSize, 0, 0, INTER_LINEAR);
                        //resize(stabilizedFrame, finalFrame, frameSize, 0, 0, INTER_LINEAR);
                        
                        // Сохраняем результат
                        //finalFrame.copyTo(frameData.frame);
                        stabilizedFrame.copyTo(frameData.frame);
                        framesStabilized++;
                    } else {
                        // Если ROI невалиден, используем полный кадр
                        stabilizedFrame.copyTo(frameData.frame);
                    }

                    // auto endTime = chrono::steady_clock::now();
                    // auto duration = chrono::duration_cast<chrono::microseconds>(endTime - startTime);
                    putText(stabilizedFrame, frameData.transform.printString(), Point(10, 30),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
                    putText(stabilizedFrame, frameData.transformFirstDerivative.printString(), Point(10, 60),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
                    imshow("stabilizedFrame", stabilizedFrame);
                    int key = waitKey(5);
                } catch (const exception& e) {
                    cerr << "Error in stabilization: " << e.what() << endl;
                    // В случае ошибки оставляем оригинальный кадр
                }
            }
            
            // if (debugMode && framesProcessed % 100 == 0) {
            //     cout << "Stabilized " << framesStabilized << "/" << framesProcessed 
            //          << " frames" << endl;
            // }
                        
            // Отправляем для отображения
            displayQueue.push(move(frameData));
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
        
        cout << "Display thread started" << endl;
        
        while (running) {
            FrameData frameData;
            if (!displayQueue.try_pop(frameData)) {
                this_thread::sleep_for(chrono::milliseconds(1));
                continue;
            }
            
            displayedFrames++;
            
            // Проверяем, что кадр не пустой
            if (frameData.frame.empty()) {
                cerr << "Empty frame in display thread" << endl;
                continue;
            }
            
            // Создаем информационный overlay
            Mat displayFrame;
            frameData.frame.copyTo(displayFrame);
            
            // Добавляем информацию о производительности
            string infoText = format("Frame: %d | FPS: %d | Points: %d | Process: %.1f ms | kSwitch: %.1f",
                                    frameData.frameId, fps.load(), trackedPoints.load(), 
                                    processingTime.load(), kSwitch);
            // string infoTransform = format("dx: %1.1f | dy: %1.1f | da: %1.1f | Process: %1.1f ms",
            //             frameData.frameId, fps.load(), trackedPoints.load(), 
            //             processingTime.load());
            putText(displayFrame, "biases" + frameData.transform.printString(), Point(10, 60),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
            putText(displayFrame, "derivarite " + frameData.transformFirstDerivative.printString(), Point(10, 90),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
            putText(displayFrame,"SKO " + frameData.transformSKO.printString(), Point(10, 120),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
            putText(displayFrame, infoText, Point(10, 30),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorGREEN, 2);
            
            // Добавляем информацию о трансформации
            string transformText = format("dX: %.1f dY: %.1f dA: %.1f deg",
                                         frameData.transform.dx, frameData.transform.dy,
                                         frameData.transform.da * RAD_TO_DEG);
            putText(displayFrame, transformText, Point(10, 60),
                   FONT_HERSHEY_SIMPLEX, 0.7, colorYELLOW, 2);
            
            
            // Рисуем точки (первые 30 для наглядности)
            if (true || !frameData.points.empty()) {
                int pointsToShow = min(200, static_cast<int>(frameData.points.size()));
                for (int i = 0; i < pointsToShow; i++) {
                    Point2f pt = frameData.points[i];
                    // Масштабируем координаты точек обратно к исходному размеру
                    Point scaledPt(static_cast<int>(pt.x * compressionConfig),
                                   static_cast<int>(pt.y * compressionConfig));
                    circle(displayFrame, scaledPt, 5, colorBLUE, -1);
                }
            }
            
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
                         << ", Queue size: " << displayQueue.size() << endl;
                }
                displayedFrames = 0;
                lastDisplayTime = now;
            }
            
            // Обработка клавиш
            int key = waitKey(1);
            if (key == 27 || key == 'q') { // ESC или Q
                cout << "Exit requested by user" << endl;
                running = false;
                break;
            } else if (key == ' ') { // Пробел - пауза
                cout << "Paused. Press any key to continue..." << endl;
                waitKey(0);
            } else if (key == 's') { // S - сохранить кадр
                string filename = format("frame_%06d.jpg", frameData.frameId);
                imwrite(filename, displayFrame);
                cout << "Frame saved: " << filename << endl;
            } else if (key == 'd') { // D - переключить режим отладки
                debugMode = !debugMode;
                cout << "Debug mode: " << (debugMode ? "ON" : "OFF") << endl;
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
    // ========================= ДЛЯ ЗАХВАТА ИЗОБРАЖЕНИЯ =========================
    void loadImage(cv::Mat& image_color, int frame_id, std::string filepath)
    {
        char file[200];
        sprintf(file, "image_0/%06d.png", frame_id);
        std::string filename = filepath + std::string(file);
        image_color = cv::imread(filename, IMREAD_COLOR);

        if (image_color.empty())
        {
            cerr << "Failed to load image: " << filename << endl;
        }
    }


    //========================= ДЛЯ ДОБАВЛЕНИЯ ХАРАКТЕРНЫХ ТОЧЕК ===================

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



    void iirAdaptive(TransformParam& transformsFirtsDerivative, TransformParam& transforms, TransformParam& transformSKO, 
        double& tau_stab, Rect& roi, const int a, const int b, double& kSwitch)
    {
        if (true ||
            (abs(transformsFirtsDerivative.dx) - 20.0 < 3.0 * transformSKO.dx) && 
            (abs(transformsFirtsDerivative.dy) - 20.0 < 3.0 * transformSKO.dy) && 
            (abs(transformsFirtsDerivative.da) - 10.0 * DEG_TO_RAD < 3.0 * transformSKO.da))
        {
            transforms.dx = kSwitch * (transforms.dx * (tau_stab - 1.0) / tau_stab + kSwitch * transformsFirtsDerivative.dx);
            transforms.dy = kSwitch * (transforms.dy * (tau_stab - 1.0) / tau_stab + kSwitch * transformsFirtsDerivative.dy);
            transforms.da = kSwitch * (transforms.da * (tau_stab - 1.0) / tau_stab + kSwitch * transformsFirtsDerivative.da);
        } 
        // else 
        // {
        //     cout << "iirAdaptive Explosion Detected" << endl;
        // }

        if (transforms.da > CV_PI)
            transforms.da -= CV_PI;
        if (transforms.da < -CV_PI)
            transforms.da += CV_PI;

        if (tau_stab < 30.0)
            tau_stab *= 1.2;

        if (tau_stab < 50.0 && !(abs(transforms.dx) > a / 2 || abs(transforms.dy) > b / 2))
            tau_stab *= 1.1;

        if (tau_stab < 100.0 && !(abs(transforms.dx) > a / 3 || abs(transforms.dy) > b / 3))
        {
            tau_stab *= 1.1;
            if (tau_stab > 100.0)
                tau_stab = 100.0;
        }

        // Проверка границ ROI
        if (roi.x + (int)transforms.dx < 0)
        {
            transforms.dx = double(1 - roi.x);
            if (tau_stab > 50) {
                tau_stab *= 0.9;
                kSwitch *= 0.95;
            }
        }
        else if (roi.x + roi.width + (int)transforms.dx >= a)
        {
            transforms.dx = (double)(a - roi.x - roi.width);
            if (tau_stab > 50) {
                tau_stab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (roi.y + (int)transforms.dy < 0)
        {
            transforms.dy = (double)(1 - roi.y);
            if (tau_stab > 10) {
                tau_stab *= 0.9;
                kSwitch *= 0.95;
            }
        }
        else if (roi.y + roi.height + (int)transforms.dy >= b)
        {
            transforms.dy = (double)(b - roi.y - roi.height);
            if (tau_stab > 50) {
                tau_stab *= 0.9;
                kSwitch *= 0.95;
            }
        }

        if (kSwitch < 1.0)
            tau_stab *= (4.0 + kSwitch) / 5.0;

        transformSKO.dx = (1.0 - 0.1) * transformSKO.dx + 0.1 * abs(transformsFirtsDerivative.dx);
        transformSKO.dy = (1.0 - 0.1) * transformSKO.dy + 0.1 * abs(transformsFirtsDerivative.dy);
        transformSKO.da = (1.0 - 0.1) * transformSKO.da + 0.1 * abs(transformsFirtsDerivative.da);

    }




    void removeClosePoints(vector<Point2f>& points, double minDistance) {
        if (points.empty()) return;
        
        // Сортируем точки по X
        sort(points.begin(), points.end(), 
             [](const Point2f& a, const Point2f& b) { return a.x < b.x; });
        
        vector<bool> toRemove(points.size(), false);
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (toRemove[i]) continue;
            
            for (size_t j = i + 1; j < points.size(); ++j) {
                if (points[j].x - points[i].x > minDistance) {
                    break;
                }
                
                float dx = points[j].x - points[i].x;
                float dy = points[j].y - points[i].y;
                float distanceSq = dx * dx + dy * dy;
                
                if (distanceSq < minDistance * minDistance) {
                    toRemove[j] = true;
                }
            }
        }
        
        // Удаляем отмеченные точки
        for (int i = static_cast<int>(points.size()) - 1; i >= 0; --i) {
            if (toRemove[i]) {
                points.erase(points.begin() + i);
            }
        }
    }
};

// ========================= ОСНОВНАЯ ФУНКЦИЯ =========================

int main() {
    cout << "========================================" << endl;
    cout << "     MULTI-THREADED VIDEO STABILIZER    " << endl;
    cout << "========================================" << endl;
    
    // Проверка поддержки OpenCL
    if (ocl::haveOpenCL()) {
        cout << "OpenCL is available" << endl;
        ocl::setUseOpenCL(true);
    } else {
        cout << "OpenCL is not available, using CPU" << endl;
    }
    
    // Создание стабилизатора
    VideoStabilizer stabilizer;
    
    // Запуск системы
    stabilizer.start();
    
    // Ожидание завершения
    cout << "Press ESC or Q in the window to exit..." << endl;
    cout << "Press SPACE to pause, S to save frame, D to toggle debug mode" << endl;
    
    // Основной цикл ожидания
   
    try {
        while (stabilizer.running)
        {
            this_thread::sleep_for(chrono::seconds(1));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    // Остановка системы
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}