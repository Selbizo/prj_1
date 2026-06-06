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
    
    cv::imwrite("test_neon_source.png", src);
    cout << "Source image saved" << endl;
    
    // Создадим матрицу для простого смещения (translate)
    // Сместим на 50 пикселей вправо и 30 вниз
    Mat M_translate = (Mat_<double>(2, 3) << 1, 0, 50,
                                            0, 1, 30);
    
    cout << "\n=== Test 1: Translation (50px right, 30px down) ===" << endl;
    cout << "Matrix M:" << endl << M_translate << endl;
    
    // Standard cv::warpAffine
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M_translate, Size(640, 480), INTER_LINEAR);
    cv::imwrite("test_neon_standard_translate.png", dst_standard);
    cout << "Standard warpAffine saved" << endl;
    
    // Our NEON version
    Mat dst_neon = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_translate, Size(640, 480), 
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_neon_neon_translate.png", dst_neon);
    cout << "NEON warpAffine saved" << endl;
    
    // Compare
    Mat diff;
    absdiff(dst_standard, dst_neon, diff);
    double max_diff = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            Vec3b v = diff.at<Vec3b>(i, j);
            max_diff = max(max_diff, (double)max({v[0], v[1], v[2]}));
        }
    }
    cout << "Max difference: " << (int)max_diff << endl;
    cout << "Frames are " << (max_diff < 5 ? "SIMILAR" : "DIFFERENT") << endl;
    
    // Now test rotation
    cout << "\n=== Test 2: Rotation (45 degrees) ===" << endl;
    
    Point2f center(320, 240);
    double angle = 45;
    Mat M_rotate = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cout << "Matrix M (rotation):" << endl << M_rotate << endl;
    
    // Standard
    Mat dst_standard_rot;
    cv::warpAffine(src, dst_standard_rot, M_rotate, Size(640, 480), INTER_LINEAR);
    cv::imwrite("test_neon_standard_rotate.png", dst_standard_rot);
    cout << "Standard warpAffine (rotation) saved" << endl;
    
    // NEON
    Mat dst_neon_rot = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon_rot, M_rotate, Size(640, 480),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_neon_neon_rotate.png", dst_neon_rot);
    cout << "NEON warpAffine (rotation) saved" << endl;
    
    // Compare
    absdiff(dst_standard_rot, dst_neon_rot, diff);
    max_diff = 0;
    for (int i = 0; i < diff.rows; i++) {
        for (int j = 0; j < diff.cols; j++) {
            Vec3b v = diff.at<Vec3b>(i, j);
            max_diff = max(max_diff, (double)max({v[0], v[1], v[2]}));
        }
    }
    cout << "Max difference: " << (int)max_diff << endl;
    cout << "Frames are " << (max_diff < 5 ? "SIMILAR" : "DIFFERENT") << endl;
    
    cout << "\nTest complete! Check PNG files for visual comparison." << endl;
    
    return 0;
}
