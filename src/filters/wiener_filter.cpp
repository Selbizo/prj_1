#include "filters/wiener_filter.h"
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

using namespace cv;
using namespace std;

// ========================= РАСЧЁТ PSF =========================
void GcalcPSF(UMat& outputImg, Size filterSize, Size psfSize, double len, double theta) {
    int scale = 4;
    Mat hCpuBig(Size(psfSize.width * scale, psfSize.height * scale), CV_32F, Scalar(0));
    Point center(psfSize.width * scale / 2, psfSize.height * scale / 2);

    Size axes(scale, cvRound(double(len * scale) * 0.5f));
    Size axes2(scale, cvRound(double(len * scale) * 0.45f));
    Size axes3(scale, cvRound(double(len * scale) * 0.3f));
    
    double angle = 90.0 - theta;

    ellipse(hCpuBig, center, axes, angle, 0, 360, Scalar(0.6), FILLED);
    ellipse(hCpuBig, center, axes2, angle, 0, 360, Scalar(0.9), FILLED);
    ellipse(hCpuBig, center, axes3, angle, 0, 360, Scalar(1.0), FILLED);
    
    Mat hCpu(psfSize, CV_32F);
    resize(hCpuBig, hCpu, psfSize, INTER_LINEAR);
    
    Mat h(filterSize, CV_32F, Scalar(0));
    hCpu(Rect(0, 0, hCpu.cols, hCpu.rows)).copyTo(
        h(Rect((filterSize.width - hCpu.cols) / 2,
               (filterSize.height - hCpu.rows) / 2,
               hCpu.cols, hCpu.rows)));

    Scalar summa = cv::sum(h);
    cv::divide(h, Scalar(summa[0]), outputImg);
}

// ========================= FFT SHIFT =========================
void Gfftshift(const cv::UMat& inputImg, cv::UMat& outputImg) {
    outputImg = inputImg.clone();
    int cx = outputImg.cols / 2;
    int cy = outputImg.rows / 2;
    
    UMat q0(outputImg, Rect(0, 0, cx, cy));
    UMat q1(outputImg, Rect(cx, 0, cx, cy));
    UMat q2(outputImg, Rect(0, cy, cx, cy));
    UMat q3(outputImg, Rect(cx, cy, cx, cy));
    UMat tmp;
    
    q0.copyTo(tmp);   q3.copyTo(q0);   tmp.copyTo(q3);
    q1.copyTo(tmp);   q2.copyTo(q1);   tmp.copyTo(q2);
}

// ========================= 2D ФИЛЬТРАЦИЯ В ЧАСТОТНОЙ ОБЛАСТИ =========================
void Gfilter2DFreq(const cv::UMat& inputImg, cv::UMat& outputImg, const cv::UMat& complexH) {
    UMat zeroMat(inputImg.size(), CV_32F, Scalar(0));
    vector<UMat> planes = { inputImg, zeroMat };
    UMat complexInput;
    merge(planes, complexInput);
    
    dft(complexInput, complexInput, DFT_COMPLEX_OUTPUT | DFT_SCALE);
    
    UMat complexOutput;
    mulSpectrums(complexInput, complexH, complexOutput, 0);
    
    dft(complexOutput, complexOutput, DFT_INVERSE | DFT_COMPLEX_INPUT);
    
    vector<UMat> planesOut;
    split(complexOutput, planesOut);
    outputImg = planesOut[0];
}

// ========================= ФИЛЬТР ВИНЕРА =========================
void GcalcWnrFilter(const cv::UMat& input_h_PSF, cv::UMat& output_G, double nsr) {
    UMat h_PSF_shifted;
    Gfftshift(input_h_PSF, h_PSF_shifted);
    
    UMat zeroMat(h_PSF_shifted.size(), CV_32F, Scalar(0));
    vector<UMat> planes = { h_PSF_shifted, zeroMat };
    UMat complexI;
    merge(planes, complexI);

    dft(complexI, complexI);
    
    vector<UMat> planesOut;
    split(complexI, planesOut);
    
    UMat denom;
    magnitude(planesOut[0], planesOut[1], denom);
    pow(denom, 2, denom);
    add(denom, nsr, denom);
    divide(planesOut[0], denom, output_G);
}

// ========================= ТАНДЕМНОЕ ОКНО =========================
void Gedgetaper(const Mat& inputImg, Mat& outputImg, double gamma, double beta) {
    int Nx = inputImg.cols;
    int Ny = inputImg.rows;
    Mat w1(1, Nx, CV_32F, Scalar(0));
    Mat w2(Ny, 1, CV_32F, Scalar(0));

    double* p1 = w1.ptr<double>(0);
    double* p2 = w2.ptr<double>(0);
    double dx = double(2.0 * CV_PI / Nx);
    double x = double(-CV_PI);
    for (int i = 0; i < Nx; i++) {
        p1[i] = double(0.5 * (tanh((x + gamma / 2) / beta) - tanh((x - gamma / 2) / beta)));
        x += dx;
    }
    double dy = double(2.0 * CV_PI / Ny);
    double y = double(-CV_PI);
    for (int i = 0; i < Ny; i++) {
        p2[i] = double(0.5 * (tanh((y + gamma / 2) / beta) - tanh((y - gamma / 2) / beta)));
        y += dy;
    }
    Mat w = w2 * w1;
    multiply(inputImg, w, outputImg);
}

// ========================= КАНАЛЬНАЯ ФИЛЬТРАЦИЯ =========================
void channelWiener(const cv::UMat* gChannel, cv::UMat* gChannelWiener, const cv::UMat* complexH) {
    Gfilter2DFreq(*gChannel, *gChannelWiener, *complexH);
}
