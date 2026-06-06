#include <opencv2/opencv.hpp>
#include <iostream>
#include <set>
#include "src/warpAffine_neon_optimized.hpp"

using namespace cv;
using namespace std;

int main() {
    // Создадим источник 640x480
    Mat src(480, 640, CV_8UC3, Scalar(50, 50, 50));
    
    // Нарисуем крест в центре
    for (int i = 200; i < 280; i++) {
        src.at<Vec3b>(i, 320) = Vec3b(0, 255, 0);  
        src.at<Vec3b>(240, i) = Vec3b(0, 0, 255);  
    }
    
    Point2f center(320, 240);
    Mat M_rotate = cv::getRotationMatrix2D(center, 45, 1.0);
    
    cout << "Testing 640x480 with rotation..." << endl;
    
    // Standard
    Mat dst_standard;
    cv::warpAffine(src, dst_standard, M_rotate, Size(640, 480), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
    
    // NEON
    Mat dst_neon = Mat::zeros(480, 640, CV_8UC3);
    WarpAffineNeonOptimized::warpAffine(src, dst_neon, M_rotate, Size(640, 480),
                                         INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0), true);
    
    // Найдем где отличаются
    Mat diff;
    absdiff(dst_standard, dst_neon, diff);
    
    set<pair<int,int>> diff_locations;
    int edge_count = 0;
    int interior_count = 0;
    const int BORDER_MARGIN = 5;
    
    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            Vec3b v = diff.at<Vec3b>(y, x);
            int d = max({v[0], v[1], v[2]});
            if (d > 5) {
                diff_locations.insert({x, y});
                
                bool is_edge = (x < BORDER_MARGIN || x >= 640-BORDER_MARGIN ||
                               y < BORDER_MARGIN || y >= 480-BORDER_MARGIN);
                if (is_edge) edge_count++;
                else interior_count++;
            }
        }
    }
    
    cout << "Total diff pixels: " << diff_locations.size() << endl;
    cout << "  Edge pixels (within " << BORDER_MARGIN << "px): " << edge_count << endl;
    cout << "  Interior pixels: " << interior_count << endl;
    
    // Показать несколько примеров из центра изображения
    cout << "\nExample differences in center area (200-280 x 200-280):" << endl;
    int shown = 0;
    for (auto& loc : diff_locations) {
        if (loc.first >= 200 && loc.first < 280 && loc.second >= 200 && loc.second < 280) {
            Vec3b v_std = dst_standard.at<Vec3b>(loc.second, loc.first);
            Vec3b v_neon = dst_neon.at<Vec3b>(loc.second, loc.first);
            cout << "  (" << loc.first << "," << loc.second << "): std=" 
                 << (int)v_std[0] << " neon=" << (int)v_neon[0] << endl;
            if (++shown >= 5) break;
        }
    }
    
    return 0;
}
