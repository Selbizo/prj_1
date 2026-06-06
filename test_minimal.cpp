#include <opencv2/opencv.hpp>
#include <iostream>
#include <iostream>
#include "src/warpAffine_neon_optimized.hpp"

using namespace cv;
using namespace std;

int main() {
    // Создадим минимальный тестовый кадр 100x100
    Mat src(100, 100, CV_8UC3, Scalar(0, 0, 0));
    
    // Нарисуем красный квадрат 20x20 в левом верхнем углу (10,10)-(30,30)
    for (int y = 10; y < 30; y++) {
        for (int x = 10; x < 30; x++) {
            src.at<Vec3b>(y, x) = Vec3b(0, 0, 255);  // красный (BGR)
        }
    }
    
    // Нарисуем зеленый квадрат 20x20 в правом нижнем углу (70,70)-(90,90)
    for (int y = 70; y < 90; y++) {
        for (int x = 70; x < 90; x++) {
            src.at<Vec3b>(y, x) = Vec3b(0, 255, 0);  // зеленый
        }
    }
    
    cv::imwrite("test_minimal_source.png", src);
    cout << "Source (minimal test): 100x100 with red square at (10,10) and green at (70,70)" << endl;
    
    // Тест 1: Идентичная трансформация (без сдвига)
    Mat M_identity = (Mat_<double>(2, 3) << 1, 0, 0, 0, 1, 0);
    
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M_identity, Size(100, 100), INTER_LINEAR);
    cv::imwrite("test_minimal_standard_identity.png", dst_standard);
    
    Mat dst_neon = Mat::zeros(100, 100, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_identity, Size(100, 100),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_minimal_neon_identity.png", dst_neon);
    
    // Сравниваем
    Mat diff;
    absdiff(dst_standard, dst_neon, diff);
    int nonzero = countNonZero(diff.reshape(1)); // count non-zero elements
    cout << "Test 1 (Identity): " << (nonzero == 0 ? "PASS" : "FAIL") << " (nonzero pixels: " << nonzero << ")" << endl;
    
    // Тест 2: Простой сдвиг на 10 пикселей вправо и вниз
    Mat M_translate = (Mat_<double>(2, 3) << 1, 0, 10, 0, 1, 10);
    
    dst_standard = Mat::zeros(100, 100, CV_8UC3);
    cv::warpAffine(src, dst_standard, M_translate, Size(100, 100), INTER_LINEAR);
    cv::imwrite("test_minimal_standard_translate.png", dst_standard);
    
    dst_neon = Mat::zeros(100, 100, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_translate, Size(100, 100),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_minimal_neon_translate.png", dst_neon);
    
    absdiff(dst_standard, dst_neon, diff);
    nonzero = countNonZero(diff.reshape(1));
    cout << "Test 2 (Translate 10,10): " << (nonzero == 0 ? "PASS" : "FAIL") << " (nonzero pixels: " << nonzero << ")" << endl;
    
    // Тест 3: Поворот на 90 градусов вокруг центра (50, 50)
    Point2f center(50, 50);
    Mat M_rotate90 = cv::getRotationMatrix2D(center, 90, 1.0);
    
    dst_standard = Mat::zeros(100, 100, CV_8UC3);
    cv::warpAffine(src, dst_standard, M_rotate90, Size(100, 100), INTER_LINEAR);
    cv::imwrite("test_minimal_standard_rotate90.png", dst_standard);
    
    dst_neon = Mat::zeros(100, 100, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_rotate90, Size(100, 100),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    cv::imwrite("test_minimal_neon_rotate90.png", dst_neon);
    
    absdiff(dst_standard, dst_neon, diff);
    nonzero = countNonZero(diff.reshape(1));
    cout << "Test 3 (Rotate 90): " << (nonzero == 0 ? "PASS" : "FAIL") << " (nonzero pixels: " << nonzero << ")" << endl;
    
    cout << "\nCheck PNG files for visual verification." << endl;
    
    return 0;
}
