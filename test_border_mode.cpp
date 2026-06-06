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
    
    Point2f center(320, 240);
    double angle = 45;
    Mat M_rotate = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cout << "Testing with BORDER_REPLICATE:" << endl;
    
    // Standard with BORDER_REPLICATE
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M_rotate, Size(640, 480), INTER_LINEAR, BORDER_REPLICATE);
    cv::imwrite("test_border_standard.png", dst_standard);
    
    // NEON with BORDER_REPLICATE (note: not supported directly - check if warpAffineNeonTileBased handles it)
    Mat dst_neon = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_rotate, Size(640, 480),
                                         INTER_LINEAR, BORDER_REPLICATE, Scalar(0, 0, 0), true);
    cv::imwrite("test_border_neon.png", dst_neon);
    
    // Compare
    Mat diff;
    absdiff(dst_standard, dst_neon, diff);
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
    cout << "Max diff=" << (int)max_diff << ", nonzero pixels=" << nonzero << endl;
    
    return 0;
}
