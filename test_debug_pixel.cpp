#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>
#include "src/warpAffine_neon_optimized.hpp"

using namespace cv;
using namespace std;

int main() {
    // Создадим источник 100x100 с квадратом в центре
    Mat src(100, 100, CV_8UC3, Scalar(100, 100, 100));
    
    // Белый квадрат 20x20 в центре (40,40)-(60,60)
    for (int y = 40; y < 60; y++) {
        for (int x = 40; x < 60; x++) {
            src.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
        }
    }
    
    // Матрица поворота на 45 градусов вокруг центра (50, 50)
    Point2f center(50, 50);
    Mat M_rotate = cv::getRotationMatrix2D(center, 45, 1.0);
    
    cout << "Testing rotation 45 degrees around (50, 50)" << endl;
    cout << "Matrix M:" << endl << M_rotate << endl << endl;
    
    // Стандартная версия
    Mat dst_standard = Mat::zeros(100, 100, CV_8UC3);
    cv::warpAffine(src, dst_standard, M_rotate, Size(100, 100), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    
    // Наша версия
    Mat dst_neon = Mat::zeros(100, 100, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_rotate, Size(100, 100),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    
    // Найдем несоответствия
    cout << "Finding first difference..." << endl;
    bool found_diff = false;
    for (int y = 0; y < 100 && !found_diff; y++) {
        for (int x = 0; x < 100 && !found_diff; x++) {
            Vec3b v_std = dst_standard.at<Vec3b>(y, x);
            Vec3b v_neon = dst_neon.at<Vec3b>(y, x);
            
            int diff = abs((int)v_std[0] - (int)v_neon[0]) +
                      abs((int)v_std[1] - (int)v_neon[1]) +
                      abs((int)v_std[2] - (int)v_neon[2]);
            
            if (diff > 2) {  // Небольшие различия в интерполяции - нормально
                cout << "First significant difference at (" << x << ", " << y << "):" << endl;
                cout << "  Standard: (" << (int)v_std[0] << ", " << (int)v_std[1] << ", " << (int)v_std[2] << ")" << endl;
                cout << "  NEON:     (" << (int)v_neon[0] << ", " << (int)v_neon[1] << ", " << (int)v_neon[2] << ")" << endl;
                cout << "  Diff:     " << diff << endl;
                found_diff = true;
            }
        }
    }
    
    if (!found_diff) {
        cout << "No significant differences found!" << endl;
    }
    
    // Сохраним результаты
    cv::imwrite("test_debug_standard.png", dst_standard);
    cv::imwrite("test_debug_neon.png", dst_neon);
    
    Mat diff_img;
    absdiff(dst_standard, dst_neon, diff_img);
    // Умножим разницы для видимости
    diff_img = diff_img * 5;
    cv::imwrite("test_debug_diff.png", diff_img);
    
    cout << "\nSaved: test_debug_standard.png, test_debug_neon.png, test_debug_diff.png" << endl;
    
    return 0;
}
