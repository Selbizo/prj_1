#include <opencv2/opencv.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <mutex>

#include <opencv2/core/ocl.hpp>
#include "basicFunctions.h"
#include "stabilizationFunctions.h"
#include "ConfigVideoStab.h"

using namespace std;
using namespace cv;

class VideoStabilizer {
private:
    // Параметры стабилизации (из ConfigVideoStab.h)
    int maxCorners;
    double qualityLevel;
    double minDistance;
    int blockSize;
    bool useHarrisDetector;
    double harrisK;
    int winSize;
    int maxLevel;
    double compression;
    //int NCoef;
    
    // Детекторы точек
    Ptr<FeatureDetector> detector;
    Ptr<FeatureDetector> detector_small;
    
    // Параметры оптического потока
    TermCriteria termcrit;
    Size winSizeLK;
    
    // Текущий и предыдущий кадры
    UMat uOldGray, uGray;
    UMat uOldCompressed, uCompressed;
    
    // Точки для отслеживания
    vector<Point2f> p0, p1;
    vector<uchar> status;
    vector<float> errFloat;
    UMat uP0;
    
    // Параметры трансформации
    vector<TransformParam> transforms;
    vector<TransformParam> movement;
    vector<TransformParam> movementKalman;
    
    // Фильтр Калмана
    KalmanFilterCV kf;
    int state_dim;
    int meas_dim;
    
    // Размеры изображения
    int a, b;
    double c;
    double atan_ba;
    
    // Параметры стабилизации
    double tauStab;
    double kSwitch;
    double framePart;
    Rect roi;
    
    // Флаги
    bool stabPossible;
    int frameCount;
    
    // Маски для поиска точек
    UMat uMaskSearch;
    UMat uMaskSearchSmall;
    UMat uMaskSearchSmallRoi;
    
    // Случайные цвета для отображения точек
    vector<Scalar> colors;
    RNG rng;
    
    // Трансформационные матрицы
    Mat T, TStab, TStabInv, TSearchPoints;
    
public:
    VideoStabilizer(int width, int height) : a(width), b(height), stabPossible(false), frameCount(0) {
        // Инициализация параметров (значения из ConfigVideoStab.h)
        maxCorners = 1000;
        qualityLevel = 0.01;
        minDistance = 10;
        blockSize = 3;
        useHarrisDetector = false;
        harrisK = 0.04;
        winSize = 21;
        maxLevel = 3;
        compression = 4.0;
        //NCoef = 2;
        
        tauStab = 100.0;
        kSwitch = 0.01;
        framePart = 0.8;
        
        // Инициализация детекторов
        detector = GFTTDetector::create(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, harrisK);
        detector_small = GFTTDetector::create(20, qualityLevel*1.5, minDistance*1.5, blockSize, useHarrisDetector, harrisK);
        
        // Параметры оптического потока
        termcrit = TermCriteria(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
        winSizeLK = Size(winSize, winSize);
        
        // Инициализация векторов
        transforms.resize(4);
        movement.resize(4);
        movementKalman.resize(4);
        
        for (int i = 0; i < 4; i++) {
            transforms[i] = {0.0, 0.0, 0.0};
            movement[i] = {0.0, 0.0, 0.0};
            movementKalman[i] = {0.0, 0.0, 0.0};
        }
        
        // Инициализация геометрических параметров
        c = sqrt(a * a + b * b);
        atan_ba = atan2(b, a);
        
        // Инициализация ROI
        roi.x = a * ((1.0 - framePart) / 2.0);
        roi.y = b * ((1.0 - framePart) / 2.0);
        roi.width = a * framePart;
        roi.height = b * framePart;
        
        // Инициализация фильтра Калмана
        state_dim = 9;
        meas_dim = 3;
        double dt = 1.0/30.0; // предполагаем 30 FPS
        
        Mat A = (Mat_<double>(state_dim, state_dim) <<
            1,  0,  dt, 0,  dt*dt/2, 0,  0,  0,  0,
            0,  1,  0,  dt, 0, dt*dt/2, 0,  0,  0,
            0,  0,  1,  0,  dt, 0,  0,  0,  0,
            0,  0,  0,  1,  0,  dt, 0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  1,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  1,  dt, dt*dt/2,
            0,  0,  0,  0,  0,  0,  0,  1,  dt,
            0,  0,  0,  0,  0,  0,  0,  0,  1);
            
        Mat C = (Mat_<double>(meas_dim, state_dim) <<
            1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0);
            
        Mat Q = Mat::eye(state_dim, state_dim, CV_64F) * 0.00001;
        Mat R = Mat::eye(meas_dim, meas_dim, CV_64F) * 10000.0;
        Mat P = Mat::eye(state_dim, state_dim, CV_64F) * 1.0;
        
        kf = KalmanFilterCV(dt, A, C, Q, R, P);
        Mat x0 = (Mat_<double>(state_dim, 1) << 0,0,0,0,0,0,0,0,0);
        kf.init(0, x0);
        
        // Инициализация масок
        Mat maskSearch = Mat::zeros(Size(a/compression, b/compression), CV_8U);
        rectangle(maskSearch, Rect(a*(1.0-0.9)/compression/2, b*(1.0-0.9)/compression/2, 
            a*0.9/compression, b*0.9/compression), Scalar(255), FILLED);
        rectangle(maskSearch, Rect(a*(1.0-0.4)/compression/2, b*(1.0-0.4)/compression/2, 
            a*0.4/compression, b*0.4/compression), Scalar(0), FILLED);
        maskSearch.copyTo(uMaskSearch);
        
        Mat maskSearchSmall = Mat::zeros(Size(a/compression, b/compression), CV_8U);
        rectangle(maskSearchSmall, Rect(a*(1.0-0.3)/compression/2, b*(1.0-0.3)/compression/2, 
            max(a,b)*0.3/compression, max(a,b)*0.3/compression), Scalar(255), FILLED);
        maskSearchSmall.copyTo(uMaskSearchSmall);
        
        // Инициализация цветов
        for (int i = 0; i < 1000; i++) {
            unsigned short blue = rng.uniform(100, 230);
            unsigned short green = rng.uniform(100, 230);
            unsigned short red = rng.uniform(100, 230);
            colors.push_back(Scalar(blue, green, red));
        }
    }
    
    // Функция инициализации первого кадра
    void initFirstFrame(const Mat& frame) {
        // Конвертируем в UMat и уменьшаем разрешение
        UMat uFrame;
        frame.copyTo(uFrame);
        
        resize(uFrame, uOldCompressed, Size(a/compression, b/compression), 0.0, 0.0, INTER_AREA);
        cvtColor(uOldCompressed, uOldGray, COLOR_BGR2GRAY);
        
        // Находим характерные точки
        vector<KeyPoint> keypoints;
        detector->detect(uOldGray, keypoints, uMaskSearch);
        
        p0.clear();
        for (const auto& kp : keypoints) {
            p0.push_back(kp.pt);
        }
        
        if (p0.size() >= maxCorners * 1/4) {
            stabPossible = true;
            convertVectorToUMat(p0, uP0);
        } else {
            stabPossible = false;
        }
        
        frameCount = 1;
    }
    
    // Функция стабилизации кадра
    Mat stabilizeFrame(const Mat& frame) {
        if (frameCount == 0) {
            initFirstFrame(frame);
            return frame.clone();
        }
        
        // Конвертируем текущий кадр
        UMat uFrame;
        frame.copyTo(uFrame);
        
        resize(uFrame, uCompressed, Size(a/compression, b/compression), 0.0, 0.0, INTER_AREA);
        cvtColor(uCompressed, uGray, COLOR_BGR2GRAY);
        
        if (!stabPossible || p0.size() < maxCorners * 1/5) {
            // Инициализация заново
            initFirstFrame(frame);
            kSwitch = 0.01;
            return frame.clone();
        }
        
        // Вычисляем оптический поток
        calcOpticalFlowPyrLK(uOldGray, uGray, p0, p1, status, errFloat, 
                            winSizeLK, maxLevel, termcrit, 0, 0.001);
        
        // Фильтруем хорошие точки
        vector<Point2f> good_new;
        Point2f meanP0(0.0f, 0.0f);
        
        for (size_t i = 0; i < p1.size(); ++i) {
            if (status[i]) {
                good_new.push_back(p1[i]);
                meanP0 += p0[i];
            }
        }
        
        if (!good_new.empty()) {
            meanP0.x /= good_new.size();
            meanP0.y /= good_new.size();
        }
        
        p0 = good_new;
        
        // Добавляем новые точки при необходимости
        if (p1.size() < double(maxCorners*5/7) && (abs(meanP0.x-a/2) < a/6 || abs(meanP0.y-b/2) < b/6)) {
            movementKalman[1].getTransformBoost(TSearchPoints, a, b, rng);
            warpAffine(uMaskSearchSmall, uMaskSearchSmallRoi, TSearchPoints, uMaskSearchSmall.size());
            addFramePoints(uGray, p0, detector_small, uMaskSearchSmallRoi);
            removeFramePoints(p0, minDistance*0.8);
        }
        
        // Вычисляем трансформацию
        Point2f d(0.0f, 0.0f);
        getBiasAndRotation(p0, p1, d, meanP0, transforms, T, compression);
        
        // Применяем адаптивную фильтрацию
        iirAdaptive(transforms, tauStab, roi, a, b, c, kSwitch, movement, movementKalman);
        
        // Обновляем фильтр Калмана
        kf.update((Mat_<double>(3, 1) << transforms[1].dx, transforms[1].dy, transforms[1].da));
        Mat state = kf.state();
        
        movementKalman[1].dx = state.at<double>(0, 0);
        movementKalman[1].dy = state.at<double>(1, 0);
        movementKalman[1].da = state.at<double>(6, 0);
        
        // Получаем матрицу стабилизации
        transforms[0].getTransform(TStab, a, b, c, atan_ba, framePart);
        
        // Применяем стабилизацию
        UMat uFrameStabilized;
        warpAffine(uFrame, uFrameStabilized, TStab, Size(a, b));
        
        // Обновляем состояния для следующего кадра
        swap(uGray, uOldGray);
        if (!p0.empty()) {
            convertVectorToUMat(p0, uP0);
        }
        
        // Адаптируем параметры
        if (kSwitch < 0.01) kSwitch = 0.01;
        if (kSwitch < 1.0) {
            kSwitch *= 1.06;
            kSwitch += 0.005;
        } else if (kSwitch > 1.0) {
            kSwitch = 1.0;
        }
        
        if (p1.size() > maxCorners*4/5) {
            maxCorners *= 1.02;
            maxCorners += 1;
            detector = GFTTDetector::create(maxCorners, qualityLevel, minDistance, blockSize, useHarrisDetector, harrisK);
        }
        
        frameCount++;
        
        // Конвертируем результат обратно в Mat
        Mat result;
        uFrameStabilized.copyTo(result);
        return result;
    }
    
    // Вспомогательные функции (должны быть определены в ConfigVideoStab.h)
    void convertVectorToUMat(const vector<Point2f>& points, UMat& uPoints) {
        Mat temp(points.size(), 2, CV_32F, (void*)points.data());
        temp.copyTo(uPoints);
    }
    
    void getBiasAndRotation(const vector<Point2f>& p0, const vector<Point2f>& p1, 
                           Point2f& d, Point2f& meanP0, vector<TransformParam>& transforms, 
                           Mat& T, double compression) {
        // Реализация вычисления смещения и вращения
        if (p0.empty() || p1.empty()) {
            transforms[1].dx = 0;
            transforms[1].dy = 0;
            transforms[1].da = 0;
            return;
        }
        
        // Вычисляем среднее смещение
        double dx_sum = 0, dy_sum = 0;
        int count = 0;
        
        for (size_t i = 0; i < p0.size(); i++) {
            if (status[i]) {
                dx_sum += (p1[i].x - p0[i].x);
                dy_sum += (p1[i].y - p0[i].y);
                count++;
            }
        }
        
        if (count > 0) {
            transforms[1].dx = dx_sum / count * compression;
            transforms[1].dy = dy_sum / count * compression;
            transforms[1].da = 0.0; // Упрощенно, без вычисления вращения
        }
    }
    
    void iirAdaptive(vector<TransformParam>& transforms, double tauStab, const Rect& roi,
                    int a, int b, double c, double kSwitch, 
                    vector<TransformParam>& movement, vector<TransformParam>& movementKalman) {
        // Упрощенная реализация адаптивной фильтрации
        double alpha = 1.0 / (1.0 + tauStab * kSwitch);
        
        transforms[0].dx = transforms[0].dx * (1 - alpha) + transforms[1].dx * alpha;
        transforms[0].dy = transforms[0].dy * (1 - alpha) + transforms[1].dy * alpha;
        transforms[0].da = transforms[0].da * (1 - alpha) + transforms[1].da * alpha;
    }
    
    void addFramePoints(const UMat& uGray, vector<Point2f>& points, 
                       const Ptr<FeatureDetector>& detector, const UMat& mask) {
        vector<KeyPoint> newKeypoints;
        detector->detect(uGray, newKeypoints, mask);
        
        for (const auto& kp : newKeypoints) {
            points.push_back(kp.pt);
        }
    }
    
    void removeFramePoints(vector<Point2f>& points, double minDist) {
        // Упрощенная реализация удаления близких точек
        vector<bool> keep(points.size(), true);
        
        for (size_t i = 0; i < points.size(); i++) {
            if (!keep[i]) continue;
            for (size_t j = i + 1; j < points.size(); j++) {
                if (!keep[j]) continue;
                double dist = norm(points[i] - points[j]);
                if (dist < minDist) {
                    keep[j] = false;
                }
            }
        }
        
        vector<Point2f> filtered;
        for (size_t i = 0; i < points.size(); i++) {
            if (keep[i]) {
                filtered.push_back(points[i]);
            }
        }
        
        points = filtered;
    }
};

class SimpleVideoProcessor {
private:
    // Очереди для передачи данных между потоками
    std::queue<cv::Mat> rawFrameQueue;
    std::queue<cv::Mat> processedFrameQueue;
    std::queue<std::pair<cv::Mat, cv::Mat>> displayQueue;
    
    // Мьютексы для каждой очереди
    std::mutex rawQueueMutex;
    std::mutex processedQueueMutex;
    std::mutex displayQueueMutex;
    
    std::atomic<bool> running{true};
    std::atomic<int> currentFrameId{0};
    std::string filepath;
    int totalFrames;
    int processedFrames = 0;
    
    // Стабилизатор видео
    VideoStabilizer* stabilizer;

public:
    SimpleVideoProcessor(const std::string& path, int startFrame = 0, int framesCount = 10000) 
        : filepath(path), currentFrameId(startFrame), totalFrames(framesCount), stabilizer(nullptr) {
        
        cv::UMat testImage;
        loadImage(testImage, startFrame, filepath);
        if (testImage.empty()) {
            std::cerr << "Не удалось загрузить начальный кадр!" << std::endl;
        }
    }
    
    ~SimpleVideoProcessor() {
        if (stabilizer) {
            delete stabilizer;
        }
    }
    
    void loadImage(cv::UMat& image_color, int frame_id, std::string filepath) {
        char file[200];
        sprintf(file, "image_0/%06d.png", frame_id);
        std::string filename = filepath + std::string(file);
        image_color = cv::imread(filename, IMREAD_COLOR).getUMat(ACCESS_READ);
        
        if (image_color.empty()) {
            cerr << "Failed to load image: " << filename << endl;
        }
    }
    
    void run() {
        // 1. Поток захвата кадров
        std::thread captureThread([this]() {
            while (running && currentFrameId.load() < totalFrames) {
                cv::UMat frame_umat;
                loadImage(frame_umat, currentFrameId.load(), filepath);
                
                if (frame_umat.empty()) {
                    running = false;
                    break;
                }
                
                cv::Mat frame = frame_umat.getMat(ACCESS_READ).clone();
                
                {
                    std::lock_guard<std::mutex> lock(rawQueueMutex);
                    if (rawFrameQueue.size() < 10) {
                        rawFrameQueue.push(frame);
                    }
                }
                
                currentFrameId++;
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            
            running = false;
            cout << "Capture thread finished." << endl;
        });
        
        // 2. Поток обработки изображений (стабилизация)
        std::thread processThread([this]() {
            while (running || !rawFrameQueue.empty()) {
                cv::Mat frame;
                
                // Извлечение кадра из очереди
                {
                    std::lock_guard<std::mutex> lock(rawQueueMutex);
                    if (!rawFrameQueue.empty()) {
                        frame = rawFrameQueue.front();
                        rawFrameQueue.pop();
                    }
                }
                
                if (frame.empty()) {
                    if (running) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(30));
                        continue;
                    } else {
                        break;
                    }
                }
                
                // Инициализация стабилизатора при первом кадре
                if (!stabilizer) {
                    stabilizer = new VideoStabilizer(frame.cols, frame.rows);
                }
                
                // Применение стабилизации
                cv::Mat stabilizedFrame = stabilizer->stabilizeFrame(frame);
                
                // Помещаем пару (оригинал + результат) в очередь отображения
                {
                    std::lock_guard<std::mutex> lock(displayQueueMutex);
                    displayQueue.push({frame.clone(), stabilizedFrame.clone()});
                }
                
                processedFrames++;
            }
            
            cout << "Processing thread finished. Processed frames: " << processedFrames << endl;
        });
        
        // 3. Поток отображения
        std::thread displayThread([this]() {
            int displayedFrames = 0;
            
            while (running || !displayQueue.empty()) {
                cv::Mat original, result;
                
                // Извлечение данных для отображения
                {
                    std::lock_guard<std::mutex> lock(displayQueueMutex);
                    if (!displayQueue.empty()) {
                        auto pair = displayQueue.front();
                        original = pair.first;
                        result = pair.second;
                        displayQueue.pop();
                        displayedFrames++;
                    }
                }
                
                if (original.empty() || result.empty()) {
                    if (running) {
                        continue;
                    } else {
                        break;
                    }
                }
                
                // Добавление информации о кадре
                string frameInfo1 = "Original Frame: " + to_string(displayedFrames);
                string frameInfo2 = "Stabilized Frame: " + to_string(displayedFrames);
                cv::putText(original, frameInfo1, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
                cv::putText(result, frameInfo2, cv::Point(10, 30), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 20, 255), 2);
                
                // Отображение
                cv::imshow("Original Video", original);
                cv::imshow("Stabilized Video", result);
                
                // Обработка нажатий клавиш
                int key = cv::waitKey(1);
                if (key == 27) { // ESC
                    running = false;
                    break;
                } else if (key == 's') {
                    cv::imwrite("saved_frame_" + to_string(displayedFrames) + ".png", result);
                    cout << "Frame saved: saved_frame_" << displayedFrames << ".png" << endl;
                } else if (key == 'p') {
                    cv::waitKey(0);
                }
            }
            
            cout << "Display thread finished. Displayed frames: " << displayedFrames << endl;
            cv::destroyAllWindows();
        });
        
        // Ожидание завершения всех потоков
        captureThread.join();
        processThread.join();
        displayThread.join();
        
        cout << "All threads finished successfully." << endl;
    }
};

int main() {
    // Укажите путь к папке с кадрами
    //string filepath = "/path/to/your/frames/folder/";
    
    // Создаем процессор
    SimpleVideoProcessor processor(filepath, 0, 10000);
    processor.run();
    
    return 0;
}