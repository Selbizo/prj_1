#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main() {
    // Создадим простую тестовую матрицу для поворота на 45 градусов
    double angle = 45;  // градусы
    double angle_rad = angle * M_PI / 180.0;
    
    // Центр поворота
    Point2f center(320, 240);
    
    // Получим матрицу поворота
    Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cout << "Original rotation matrix M (source -> target):" << endl;
    cout << M << endl << endl;
    
    // Теперь инвертируем как в нашем коде
    Mat M_inv = Mat::eye(2, 3, CV_64F);
    
    Mat A = M(cv::Rect(0, 0, 2, 2)).clone();
    Mat A_inv = A.inv();
    Mat t = M(cv::Rect(2, 0, 1, 2)).clone();
    Mat t_inv = -A_inv * t;
    
    A_inv.copyTo(M_inv(cv::Rect(0, 0, 2, 2)));
    t_inv.copyTo(M_inv(cv::Rect(2, 0, 1, 2)));
    
    cout << "Inverted matrix M_inv (target -> source):" << endl;
    cout << M_inv << endl << endl;
    
    // Тест: возьмем точку в целевом пространстве и проверим координаты в исходном
    // Например, центр (320, 240) должна оставаться неизменной
    float dst_x = 320, dst_y = 240;
    float src_x = M_inv.at<double>(0, 0) * dst_x + M_inv.at<double>(0, 1) * dst_y + M_inv.at<double>(0, 2);
    float src_y = M_inv.at<double>(1, 0) * dst_x + M_inv.at<double>(1, 1) * dst_y + M_inv.at<double>(1, 2);
    
    cout << "Testing inversion:" << endl;
    cout << "Target (dst) point: (" << dst_x << ", " << dst_y << ")" << endl;
    cout << "Source (src) point: (" << src_x << ", " << src_y << ")" << endl;
    cout << "Expected: (~320, ~240)" << endl << endl;
    
    // Создадим источник 640x480 с простым паттерном (диагональ)
    Mat src(480, 640, CV_8UC3, Scalar(0, 0, 0));
    
    // Нарисуем диагональ от верхнего левого к нижнему правому углу
    for (int i = 0; i < 480; i++) {
        for (int j = 0; j < 640; j++) {
            if (i == j / 2) {
                src.at<Vec3b>(i, j) = Vec3b(255, 255, 255);  // белые пиксели на диагонали
            }
        }
    }
    
    // Нарисуем крест в центре
    for (int i = 200; i < 280; i++) {
        src.at<Vec3b>(i, 320) = Vec3b(0, 255, 0);  // зеленая вертикальная линия
        src.at<Vec3b>(240, i) = Vec3b(0, 0, 255);  // красная горизонтальная линия
    }
    
    // Применим стандартное cv::warpAffine с исходной матрицей
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M, Size(640, 480), cv::INTER_LINEAR);
    
    cout << "Applied cv::warpAffine with original matrix M" << endl;
    cout << "Saved to: test_warp_standard.png" << endl;
    cv::imwrite("test_warp_standard.png", dst_standard);
    
    // Также сохраним исходное изображение
    cv::imwrite("test_warp_source.png", src);
    
    return 0;
}
