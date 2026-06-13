#include "stabilizationFunctions.h"
#include "basicFunctions.h"

using namespace cv;
using namespace std;

// Функция создания детекторов (CPU версии)
void createDetectors(Ptr<FeatureDetector>& detector, Ptr<FeatureDetector>& detector_small,
    Ptr<FeatureDetector>& detector_extra)
{
    detector = GFTTDetector::create(
        maxCornersConfig, qualityLevelConfig, minDistanceConfig, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
        
    detector_small = GFTTDetector::create(
        20, qualityLevelConfig*1.5, minDistanceConfig*1.5, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
    
    detector_extra = GFTTDetector::create(
        50, qualityLevelConfig*0.8, minDistanceConfig*0.8, blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
}

// Функция инициализации первого кадра
void initFirstFrame(bool cameraInUse, VideoCapture& capture, string filepath, 
    int frame_id, Mat& oldFrame, UMat& uOldFrame, UMat& uOldCompressed, 
    UMat& uOldGray, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible)
{
    if (cameraInUse)
    {
        capture >> uOldFrame;
    }
    else
    {
        loadImage(uOldFrame, frame_id, filepath);
    }
    
    //UMat uMattemp;
    
    //oldFrame.copyTo(uOldFrame);
    
    resize(uOldFrame, uOldCompressed, Size(a / compression, b / compression), 
           0.0, 0.0, INTER_LINEAR);
    // cvtColor(uOldCompressed, uMattemp, COLOR_BGR2GRAY);
    cvtColor(uOldCompressed, uOldGray, COLOR_BGR2GRAY);
    
    // Применяем билатеральный фильтр (CPU версия)
    // Mat oldGrayMat;
    //uOldGray.copyTo(oldGrayMat);
    // bilateralFilter(uMattemp, uOldGray, 3, 3.0, 1.0);
    //oldGrayMat.copyTo(uOldGray);

    if (qualityLevel > 0.001 && harrisK > 0.001)
    {
        qualityLevel *= 0.6;
        harrisK *= 0.6;
    }
    else
    {
        if (maxCorners > 50)
        {
            maxCorners *= 0.98;
            detector = GFTTDetector::create(maxCorners, qualityLevel, minDistanceConfig,
                                           blockSizeConfig, useHarrisDetectorConfig, harrisKConfig);
        }
    }
    
    for (int i = 0; i < 1; i++)
    {
        transforms[i].dx *= kSwitch;
        transforms[i].dy *= kSwitch;
        transforms[i].da *= kSwitch;
    }

    // Обнаружение точек на CPU
    //Mat maskMat = mask_device.getMat(ACCESS_READ);
    
    vector<KeyPoint> keypoints;
    // detector->detect(oldGrayMat, keypoints, maskMat);
    detector->detect(uOldGray, keypoints, mask_device);
    
    // Конвертируем KeyPoint в Point2f
    p0.clear();
    for (const auto& kp : keypoints)
    {
        p0.push_back(kp.pt);
    }
    
    // Загружаем точки в UMat
    if (!p0.empty())
    {
        convertVectorToUMat(p0,uP0);
    }

    stab_possible = (p0.size() > 6);
}

void initFirstFrame(UMat& uOldGray, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible)
{
    if (qualityLevel > 0.001 && harrisK > 0.001)
    {
        qualityLevel *= 0.6;
        harrisK *= 0.6;
    }
    else
    {
        if (maxCorners > 50)
        {
            maxCorners *= 0.98;
            detector = GFTTDetector::create(maxCorners, qualityLevel, minDistanceConfig,
                                           blockSizeConfig, useHarrisDetectorConfig, harrisK);
        }
    }
    
    for (int i = 0; i < 1; i++)
    {
        transforms[i].dx *= kSwitch;
        transforms[i].dy *= kSwitch;
        transforms[i].da *= kSwitch;
    }

    // Обнаружение точек на CPU
    Mat oldGrayMat;
    uOldGray.copyTo(oldGrayMat);
    
    Mat maskMat;
    mask_device.copyTo(maskMat);
    
    vector<KeyPoint> keypoints;
    detector->detect(oldGrayMat, keypoints, maskMat);
    
    // Конвертируем KeyPoint в Point2f
    p0.clear();
    for (const auto& kp : keypoints)
    {
        p0.push_back(kp.pt);
    }
    
    // Загружаем точки в UMat
    if (!p0.empty())
    {
        Mat(p0).copyTo(uP0.getMat(ACCESS_WRITE));
    }

    stab_possible = (p0.size() > 20);
}

void initFirstFrameZero(Mat& oldFrame, UMat& uOldFrame, UMat& uOldGray,
    UMat& uOldCompressed, UMat& uP0, vector<Point2f>& p0,
    double& qualityLevel, double& harrisK, int& maxCorners, 
    Ptr<FeatureDetector>& detector, vector<TransformParam>& transforms,
    double& kSwitch, const int a, const int b, const int compression, 
    UMat& mask_device, bool& stab_possible)
{
    //oldFrame.copyTo(uOldFrame);
    //UMat uMattemp(Size(a,b), CV_8UC1);
    resize(uOldFrame, uOldCompressed, Size(a / compression, b / compression), 
           0.0, 0.0, INTER_AREA);
    cvtColor(uOldCompressed, uOldGray, COLOR_BGR2GRAY);
        
    //bilateralFilter(uMattemp, uOldGray, 3, 3.0, 3.0);
    stab_possible = false;
}

void getBiasAndRotation(vector<Point2f>& p0, vector<Point2f>& p1, Point2f& d, 
    Point2f& meanP0, vector<TransformParam>& transforms, Mat& T, const int compression)
{
    const double N = 1.0;
    
    if (p0.empty() || p1.empty() || p0.size() != p1.size())
    {
        d = Point2f(0, 0);
        meanP0 = Point2f(0, 0);
        transforms[1] = TransformParam(0.0, 0.0, 0.0);
        cout << "Warning: Empty or mismatched point sets" << endl;
        return;
    }
    
    for (uint i = 0; i < p1.size(); i++)
    {
        if (i == 0)
        {
            d = p1[0] - p0[0];
            meanP0 = p1[0];
        }
        d = d + (p1[i] - p0[i]);
        meanP0 = meanP0 + p1[i];
    }

    d = d * compression / (int)p0.size();
    meanP0 = meanP0 * compression / (int)p0.size();

    T = estimateAffine2D(p0, p1);
    transforms[1] = TransformParam(
        -(T.at<double>(0, 2) * N + d.x * (1.0 - N)) * compression,
        -(T.at<double>(1, 2) * N + d.y * (1.0 - N)) * compression,
        -atan2(T.at<double>(1, 0), T.at<double>(0, 0)));
}

// Функция добавления точек кадра (CPU версия)
void addFramePoints(UMat& uOldGray, vector<Point2f>& p0,
    Ptr<FeatureDetector>& detector, UMat& uMaskSearchSmall)
{
    try {
        vector<KeyPoint> keypoints;
        
        if (detector.empty() || uOldGray.empty()) {
            return;
        }

        // Скачиваем данные для обработки на CPU
        Mat oldGrayMat;
        uOldGray.copyTo(oldGrayMat);
        
        Mat maskMat;
        if (!uMaskSearchSmall.empty()) {
            uMaskSearchSmall.copyTo(maskMat);
        }

        // Обнаружение точек
        detector->detect(oldGrayMat, keypoints, maskMat);

        // Добавление точек
        if (!keypoints.empty()) {
            for (const auto& kp : keypoints) {
                p0.push_back(kp.pt);
            }
        }
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV exception in addFramePoints: " << e.what() << endl;
    }
    catch (const exception& e) {
        cerr << "Standard exception in addFramePoints: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown exception in addFramePoints" << endl;
    }
}

void addFramePoints(UMat& uOldGray, vector<Point2f>& p0,
    Ptr<FeatureDetector>& detector)
{
    try {
        vector<KeyPoint> keypoints;
        
        if (detector.empty() || uOldGray.empty()) {
            return;
        }

        Mat oldGrayMat;
        uOldGray.copyTo(oldGrayMat);

        detector->detect(oldGrayMat, keypoints);

        if (!keypoints.empty()) {
            for (const auto& kp : keypoints) {
                p0.push_back(kp.pt);
            }
        }
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV exception in addFramePoints: " << e.what() << endl;
    }
    catch (const exception& e) {
        cerr << "Standard exception in addFramePoints: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown exception in addFramePoints" << endl;
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

void iirAdaptiveOld(vector<TransformParam>& transforms, double& tau_stab, 
    Rect& roi, const int a, const int b, const double c, double& kSwitch)
{
    if ((abs(transforms[1].dx) - 20.0 < 4.0 * transforms[3].dx) && 
        (abs(transforms[1].dy) - 20.0 < 4.0 * transforms[3].dy) && 
        (abs(transforms[1].da) - 0.04 < 4.0 * transforms[3].da))
    {
        transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx);
        transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy);
        transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da);
    } 
    else 
    {
        cout << "iirAdaptiveException" << endl;
    }

    if (transforms[0].da > CV_PI)
        transforms[0].da -= CV_PI;
    if (transforms[0].da < -CV_PI)
        transforms[0].da += CV_PI;

    if (tau_stab < 30.0)
        tau_stab *= 1.1;

    if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
        tau_stab *= 1.1;

    if (tau_stab < 500.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
    {
        tau_stab *= 1.1;
        if (tau_stab > 500.0)
            tau_stab = 500.0;
    }

    // Проверка границ ROI
    if (roi.x + (int)transforms[0].dx < 0)
    {
        transforms[0].dx = double(1 - roi.x);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            transforms[0].da *= 0.999;
            kSwitch *= 0.95;
        }
    }
    else if (roi.x + roi.width + (int)transforms[0].dx >= a)
    {
        transforms[0].dx = (double)(a - roi.x - roi.width);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            transforms[0].da *= 0.999;
            kSwitch *= 0.95;
        }
    }

    if (roi.y + (int)transforms[1].dy < 0)
    {
        transforms[0].dy = (double)(1 - roi.y);
        if (tau_stab > 10) {
            tau_stab *= 0.9;
            transforms[0].da *= 0.999;
            kSwitch *= 0.95;
        }
    }
    else if (roi.y + roi.height + (int)transforms[0].dy >= b)
    {
        transforms[0].dy = (double)(b - roi.y - roi.height);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            transforms[0].da *= 0.999;
            kSwitch *= 0.95;
        }
    }

    if (kSwitch < 1.0)
        tau_stab *= (4.0 + kSwitch) / 5.0;

    transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx);
    transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy);
    transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da);

    transforms[2].dx = 0.0;
    transforms[2].dy = 0.0;
    transforms[2].da = 0.0;
}

void iirAdaptiveHighPass(vector<TransformParam>& transforms, double& tau_stab, 
    Rect& roi, const int a, const int b, const double c, double& kSwitch, 
    vector<TransformParam>& movement, vector<TransformParam>& movementKalman)
{
    if ((abs(transforms[1].dx) - 20.0 < 3.0 * transforms[3].dx) && 
        (abs(transforms[1].dy) - 20.0 < 3.0 * transforms[3].dy) && 
        (abs(transforms[1].da) - 10.0 * DEG_TO_RAD < 3.0 * transforms[3].da))
    {
        transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx) - movementKalman[1].dx;
        transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy) - movementKalman[1].dy;
        transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da) - movementKalman[1].da;
    } 
    else 
    {
        cout << "iirAdaptiveHighPass Explosion Detected" << endl;
    }

    if (transforms[0].da > CV_PI)
        transforms[0].da -= CV_PI;
    if (transforms[0].da < -CV_PI)
        transforms[0].da += CV_PI;

    if (tau_stab < 30.0)
        tau_stab *= 1.2;

    if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
        tau_stab *= 1.1;

    if (tau_stab < 100.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
    {
        tau_stab *= 1.1;
        if (tau_stab > 100.0)
            tau_stab = 100.0;
    }

    // Проверка границ ROI
    if (roi.x + (int)transforms[0].dx < 0)
    {
        transforms[0].dx = double(1 - roi.x);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }
    else if (roi.x + roi.width + (int)transforms[0].dx >= a)
    {
        transforms[0].dx = (double)(a - roi.x - roi.width);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (roi.y + (int)transforms[1].dy < 0)
    {
        transforms[0].dy = (double)(1 - roi.y);
        if (tau_stab > 10) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }
    else if (roi.y + roi.height + (int)transforms[0].dy >= b)
    {
        transforms[0].dy = (double)(b - roi.y - roi.height);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (kSwitch < 1.0)
        tau_stab *= (4.0 + kSwitch) / 5.0;

    transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx - movementKalman[1].dx);
    transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy - movementKalman[1].dy);
    transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da - movementKalman[1].da);

    double moveTau = 0.8;
    movement[1].dy = movement[1].dy * moveTau + transforms[0].dy * (1.0 - moveTau);
    movement[1].da = movement[1].da * moveTau + transforms[0].da * (1.0 - moveTau);
    movement[1].dx = movement[1].dx * moveTau + transforms[0].dx * (1.0 - moveTau);

    movement[0].dy = movement[1].dy + movement[0].dy * 0.95;
    movement[0].da = movement[1].da + movement[0].da * 0.95;
    movement[0].dx = movement[1].dx + movement[0].dx * 0.95;

    movementKalman[0].dy = movementKalman[1].dy + movementKalman[0].dy * 0.988;
    movementKalman[0].da = movementKalman[1].da + movementKalman[0].da * 0.988;
    movementKalman[0].dx = movementKalman[1].dx + movementKalman[0].dx * 0.988;

    transforms[2].dx = 0.0;
    transforms[2].dy = 0.0;
    transforms[2].da = 0.0;
}

void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, 
    Rect& roi, const int a, const int b, const double c, double& kSwitch, 
    vector<TransformParam>& movement, vector<TransformParam>& movementKalman)
{
    if ((abs(transforms[1].dx) - 20.0 < 3.0 * transforms[3].dx) && 
        (abs(transforms[1].dy) - 20.0 < 3.0 * transforms[3].dy) && 
        (abs(transforms[1].da) - 10.0 * DEG_TO_RAD < 3.0 * transforms[3].da))
    {
        transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx);
        transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy);
        transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da);
    } 
    else 
    {
        cout << "iirAdaptiveHighPass Explosion Detected" << endl;
    }

    if (transforms[0].da > CV_PI)
        transforms[0].da -= CV_PI;
    if (transforms[0].da < -CV_PI)
        transforms[0].da += CV_PI;

    if (tau_stab < 30.0)
        tau_stab *= 1.2;

    if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
        tau_stab *= 1.1;

    if (tau_stab < 100.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3))
    {
        tau_stab *= 1.1;
        if (tau_stab > 100.0)
            tau_stab = 100.0;
    }

    // Проверка границ ROI
    if (roi.x + (int)transforms[0].dx < 0)
    {
        transforms[0].dx = double(1 - roi.x);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }
    else if (roi.x + roi.width + (int)transforms[0].dx >= a)
    {
        transforms[0].dx = (double)(a - roi.x - roi.width);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (roi.y + (int)transforms[1].dy < 0)
    {
        transforms[0].dy = (double)(1 - roi.y);
        if (tau_stab > 10) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }
    else if (roi.y + roi.height + (int)transforms[0].dy >= b)
    {
        transforms[0].dy = (double)(b - roi.y - roi.height);
        if (tau_stab > 50) {
            tau_stab *= 0.9;
            kSwitch *= 0.95;
        }
    }

    if (kSwitch < 1.0)
        tau_stab *= (4.0 + kSwitch) / 5.0;

    transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx);
    transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy);
    transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da);

    double moveTau = 0.8;
    movement[1].dy = movement[1].dy * moveTau + transforms[0].dy * (1.0 - moveTau);
    movement[1].da = movement[1].da * moveTau + transforms[0].da * (1.0 - moveTau);
    movement[1].dx = movement[1].dx * moveTau + transforms[0].dx * (1.0 - moveTau);

    movement[0].dy = movement[1].dy + movement[0].dy * 0.95;
    movement[0].da = movement[1].da + movement[0].da * 0.95;
    movement[0].dx = movement[1].dx + movement[0].dx * 0.95;

    movementKalman[0].dy = movementKalman[1].dy + movementKalman[0].dy * 0.988;
    movementKalman[0].da = movementKalman[1].da + movementKalman[0].da * 0.988;
    movementKalman[0].dx = movementKalman[1].dx + movementKalman[0].dx * 0.988;
}

/*
void iirAdaptive(vector<TransformParam>& transforms, double& tau_stab, 
                 Rect& roi, const int a, const int b, const double c, double& kSwitch, 
                 vector<TransformParam>& movement, vector<TransformParam>& movementKalman) {
    // Постоянные константы
    constexpr double MAX_DISPLACEMENT_X = 20.0;
    constexpr double MAX_DISPLACEMENT_Y = 20.0;
    constexpr double MAX_ROTATION_DEGREES = 5.0;
    constexpr double SAFE_THRESHOLD_X = 10.0;
    constexpr double SAFE_THRESHOLD_Y = 10.0;
    constexpr double SAFE_THRESHOLD_DEGREES = 3.0;
    constexpr double REDUCTION_FACTOR = 0.9;

    // Обработка стабилизационных сдвигов
    if (abs(transforms[1].dx) < MAX_DISPLACEMENT_X &&
        abs(transforms[1].dy) < MAX_DISPLACEMENT_Y &&
        abs(transforms[1].da) < MAX_ROTATION_DEGREES * M_PI / 180.0) {
        transforms[0].dx = kSwitch * (transforms[0].dx * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dx);
        transforms[0].dy = kSwitch * (transforms[0].dy * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].dy);
        transforms[0].da = kSwitch * (transforms[0].da * (tau_stab - 1.0) / tau_stab + kSwitch * transforms[1].da);
    } else {
        cout << "iirAdaptive: Outlier detected." << endl;
    }

    // Нормализация угла поворота
    if (transforms[0].da > CV_PI)
        transforms[0].da -= CV_PI;
    if (transforms[0].da < -CV_PI)
        transforms[0].da += CV_PI;

    // Регулировка постоянной адаптации
    if (tau_stab < 30.0)
        tau_stab *= 1.2;

    if (tau_stab < 50.0 && !(abs(transforms[0].dx) > a / 2 || abs(transforms[0].dy) > b / 2))
        tau_stab *= 1.1;

    if (tau_stab < 100.0 && !(abs(transforms[0].dx) > a / 3 || abs(transforms[0].dy) > b / 3)) {
        tau_stab *= 1.1;
        if (tau_stab > 100.0)
            tau_stab = 100.0;
    }

    // Контроль краевых эффектов
    if (roi.x + transforms[0].dx < 0 ||
        roi.x + roi.width + transforms[0].dx >= a ||
        roi.y + transforms[0].dy < 0 ||
        roi.y + roi.height + transforms[0].dy >= b) {
        transforms[0].dx = clamp(transforms[0].dx, (double)(-roi.x), (double)(a - roi.x - roi.width));
        transforms[0].dy = clamp(transforms[0].dy, (double)(-roi.y), (double)(b - roi.y - roi.height));
    }

    // Подавляем большие колебания
    if (abs(transforms[1].dx) > SAFE_THRESHOLD_X ||
        abs(transforms[1].dy) > SAFE_THRESHOLD_Y ||
        abs(transforms[1].da) > SAFE_THRESHOLD_DEGREES * M_PI / 180.0) {
        kSwitch *= REDUCTION_FACTOR;
    }

    // Сохранение предыдущих значений для анализа динамики
    transforms[3].dx = (1.0 - 0.1) * transforms[3].dx + 0.1 * abs(transforms[1].dx);
    transforms[3].dy = (1.0 - 0.1) * transforms[3].dy + 0.1 * abs(transforms[1].dy);
    transforms[3].da = (1.0 - 0.1) * transforms[3].da + 0.1 * abs(transforms[1].da);

    // Поддерживаем нулевые перемещения по умолчанию
    transforms[2].dx = 0.0;
    transforms[2].dy = 0.0;
    transforms[2].da = 0.0;
}

*/

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

void loadImage(cv::UMat& image_color, int frame_id, std::string filepath)
{
    char file[200];
    sprintf(file, "image_0/%06d.png", frame_id);
    std::string filename = filepath + std::string(file);
    image_color = cv::imread(filename, IMREAD_COLOR).getUMat(ACCESS_READ);
    
    if (image_color.empty())
    {
        cerr << "Failed to load image: " << filename << endl;
    }
}


void addGaussianNoise(cv::Mat& image, double mean, double stddev)
{
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, mean, stddev);
    image += noise;
}