#pragma once
#include "config/config.h"
#include <opencv2/core/ocl.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <string>
#include <chrono>

// ========================= СТРУКТУРА ТРАНСФОРМАЦИИ =========================
struct TransformParam {
    double dx = 0, dy = 0, da = 0;

    TransformParam() = default;
    TransformParam(double _dx, double _dy, double _da) : dx(_dx), dy(_dy), da(_da) {}

    void getTransform(UMat& T) const;
    void getTransformInvert(UMat& T) const;
    // Обратное преобразование сразу в координатах ROI: dst_roi(x,y) = src_full(M*[x+roi.x, y+roi.y]).
    // Позволяет делать warpAffine сразу в размер ROI (меньше пикселей -> быстрее).
    void getTransformInvertROI(Mat& T, const Rect& roi) const;
    void print() const;
    string printString() const;
};

// ========================= СТРУКТУРА ДАННЫХ КАДРА =========================
struct FrameData {
    Mat frameCPU;
    Mat grayCPU;
    vector<Point2f> points;
    UMat frameGPU;
    UMat grayGPU;

    TransformParam transformSKO;
    TransformParam transformFirstDerivative;
    TransformParam transform;
    Mat stabMatrix;   // 2x3 CV_64F, уже с учётом смещения ROI (warp сразу в область ROI)

    int frameId = 0;
    double timestamp = 0;
    bool shouldSkip = false;
};

// ========================= СТРУКТУРА ВОЗМОЖНОСТЕЙ GPU =========================
struct GPUCapabilities {
    bool supportsFP16 = false;
    bool supportsHalfType = false;
    string gpuName;
};

// ========================= ФУНКЦИИ TRANSFORMPARAM =========================
inline void TransformParam::getTransform(UMat& T) const {
    T = UMat::zeros(2, 3, CV_64F);
    Mat T_cpu = T.getMat(ACCESS_WRITE);
    T_cpu.at<double>(0, 0) = cos(da);  T_cpu.at<double>(0, 1) = -sin(da);  T_cpu.at<double>(0, 2) = dx;
    T_cpu.at<double>(1, 0) = sin(da);  T_cpu.at<double>(1, 1) = cos(da);   T_cpu.at<double>(1, 2) = dy;
}

inline void TransformParam::getTransformInvert(UMat& T) const {
    T = UMat::zeros(2, 3, CV_64F);
    Mat T_cpu = T.getMat(ACCESS_WRITE);
    T_cpu.at<double>(0, 0) = cos(da);  T_cpu.at<double>(0, 1) = sin(da);   T_cpu.at<double>(0, 2) = -dx;
    T_cpu.at<double>(1, 0) = -sin(da); T_cpu.at<double>(1, 1) = cos(da);   T_cpu.at<double>(1, 2) = -dy;
}

inline void TransformParam::getTransformInvertROI(Mat& T, const Rect& roi) const {
    if (T.rows != 2 || T.cols != 3 || T.type() != CV_64F)
        T = Mat(2, 3, CV_64F);
    // warpAffine() без флага WARP_INVERSE_MAP инвертирует переданную матрицу сам.
    // Сдвиг выходного окна на roi.(x,y) соответствует вычитанию (roi.x, roi.y)
    // из вектора переноса. Это даёт ровно тот же результат, что warp полного кадра + crop(roi).
    T.at<double>(0, 0) = cos(da);  T.at<double>(0, 1) = sin(da);
    T.at<double>(0, 2) = -dx - roi.x;
    T.at<double>(1, 0) = -sin(da); T.at<double>(1, 1) = cos(da);
    T.at<double>(1, 2) = -dy - roi.y;
}

inline void TransformParam::print() const {
    cout << "TransformPrint: dx=" << dx << " dy=" << dy << " da=" << da * RAD_TO_DEG << " deg" << endl;
}

inline string TransformParam::printString() const {
    return "Transform: dx= " + to_string(dx) + " dy= " + to_string(dy) + " da= " + to_string(da) + "\n";
}

// ========================= ДЕТЕКЦИЯ GPU =========================
GPUCapabilities detectGPUCapabilities();

// ========================= WARP AFFINE ОПТИМИЗИРОВАННЫЙ =========================
void warpAffineOptimized(InputArray src, OutputArray dst, InputArray M, Size dsize,
                         int flags, int borderMode, const Scalar& borderValue, bool useFP16);
