#include <opencv2/opencv.hpp>
#include <iostream>
#include "src/warpAffine_neon_optimized.hpp"

using namespace cv;
using namespace std;

int main() {
    // Создадим источник 640x480 с простым паттерном
    Mat src(480, 640, CV_8UC3, Scalar(50, 50, 50));
    
    // Нарисуем крест в центре
    for (int i = 200; i < 280; i++) {
        src.at<Vec3b>(i, 320) = Vec3b(0, 255, 0);  // зеленая вертикальная линия
        src.at<Vec3b>(240, i) = Vec3b(0, 0, 255);  // красная горизонтальная линия
    }
    
    // Нарисуем диагональ
    for (int i = 0; i < 480; i++) {
        if (i / 2 < 640) {
            src.at<Vec3b>(i, i / 2) = Vec3b(255, 255, 0);  // голубая диагональ
        }
    }
    
    // Test ONLY rotation 45 degrees
    cout << "\n=== Testing Rotation 45 degrees ===" << endl;
    
    Point2f center(320, 240);
    double angle = 45;
    Mat M_rotate = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cout << "Matrix M (rotation):" << endl << M_rotate << endl;
    
    // Standard with INTER_NEAREST
    Mat dst_standard_nearest;
    cv::warpAffine(src, dst_standard_nearest, M_rotate, Size(640, 480), INTER_NEAREST);
    cv::imwrite("test_rotate45_standard_nearest.png", dst_standard_nearest);
    
    // NEON with INTER_NEAREST
    Mat dst_neon_nearest = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon_nearest, M_rotate, Size(640, 480),
                                         INTER_NEAREST, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_rotate45_neon_nearest.png", dst_neon_nearest);
    
    // Compare
    Mat diff;
    absdiff(dst_standard_nearest, dst_neon_nearest, diff);
    double max_diff = 0;
    int nonzero = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            Vec3b v = diff.at<Vec3b>(i, j);
            int d = max({v[0], v[1], v[2]});
            if (d > 0) nonzero++;
            max_diff = max(max_diff, (double)d);
        }
    }
    cout << "INTER_NEAREST: Max diff=" << (int)max_diff << ", nonzero pixels=" << nonzero << endl;
    
    // Standard with INTER_LINEAR
    Mat dst_standard_linear;
    cv::warpAffine(src, dst_standard_linear, M_rotate, Size(640, 480), INTER_LINEAR);
    cv::imwrite("test_rotate45_standard_linear.png", dst_standard_linear);
    
    // NEON with INTER_LINEAR
    Mat dst_neon_linear = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon_linear, M_rotate, Size(640, 480),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_rotate45_neon_linear.png", dst_neon_linear);
    
    // Compare
    absdiff(dst_standard_linear, dst_neon_linear, diff);
    max_diff = 0;
    nonzero = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            Vec3b v = diff.at<Vec3b>(i, j);
            int d = max({v[0], v[1], v[2]});
            if (d > 0) nonzero++;
            max_diff = max(max_diff, (double)d);
        }
    }
    cout << "INTER_LINEAR: Max diff=" << (int)max_diff << ", nonzero pixels=" << nonzero << endl;
    
    cout << "\nTest complete! Check PNG files." << endl;
    
    return 0;
}
