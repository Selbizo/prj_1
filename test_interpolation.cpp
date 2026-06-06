#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

int main() {
    // Создадим очень маленькое 4x4 изображение со знакомыми значениями
    Mat src(4, 4, CV_8UC1, Scalar(100));
    
    // Устанавливаем уникальные значения
    src.at<uint8_t>(0, 0) = 10;   // top-left
    src.at<uint8_t>(0, 1) = 20;   // top-middle
    src.at<uint8_t>(1, 0) = 30;   // middle-left
    src.at<uint8_t>(1, 1) = 40;   // middle-middle (квадрат для интерполяции)
    
    cout << "Source 4x4:" << endl;
    cout << src << endl << endl;
    
    // Теперь давайте интерполируем в середину квадрата (0.5, 0.5)
    // Это должно быть: 0.25 * 10 + 0.25 * 20 + 0.25 * 30 + 0.25 * 40 = 25
    
    // Используем cv::warpAffine для простого смещения + масштабирования
    // Для тестирования используем масштабирование в 2x на 8x8
    Mat M = (Mat_<double>(2, 3) << 0.5, 0, 0, 0, 0.5, 0);
    
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M, Size(8, 8), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    
    cout << "After warpAffine (scale 0.5x to 8x8):" << endl;
    cout << dst_standard << endl << endl;
    
    // Теперь создадим тест где мы  знаем точные значения
    Mat src2(2, 2, CV_8UC1);
    src2.at<uint8_t>(0, 0) = 0;
    src2.at<uint8_t>(0, 1) = 100;
    src2.at<uint8_t>(1, 0) = 100;
    src2.at<uint8_t>(1, 1) = 200;
    
    cout << "Source 2x2 (для проверки интерполяции):" << endl;
    cout << src2 << endl << endl;
    
    // Масштабируем в 4x4 (каждый пиксель становится 2x2)
    Mat M2 = (Mat_<double>(2, 3) << 0.5, 0, 0, 0, 0.5, 0);
    Mat dst2;
    cv::warpAffine(src2, dst2, M2, Size(4, 4), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));
    
    cout << "Result after scaling to 4x4:" << endl;
    cout << dst2 << endl << endl;
    
    // Точные значения для (0,0) -> 4x4:
    // (0, 0) -> (0, 0) = src2(0,0) = 0
    // (1, 0) -> (2, 0) = src2(0, 1) = 100
    // (0, 1) -> (0, 2) = src2(1, 0) = 100
    // (1, 1) -> (2, 2) = src2(1, 1) = 200
    
    // (0.5, 0.5) -> (1, 1) должно быть интерполяцией всех 4 углов
    // = 0.25 * (0 + 100 + 100 + 200) = 100
    
    cout << "Expected at (1,1): 100 (from interpolation of 4 corners)" << endl;
    cout << "Actual at (1,1): " << (int)dst2.at<uint8_t>(1, 1) << endl;
    
    return 0;
}
