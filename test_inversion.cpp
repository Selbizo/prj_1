#include <opencv2/opencv.hpp>
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

int main() {
    // Создадим матрицу для поворота на 45 градусов вокруг центра (320, 240)
    Point2f center(320, 240);
    double angle = 45;
    Mat M = cv::getRotationMatrix2D(center, angle, 1.0);
    
    cout << "Original matrix M (source -> target):" << endl;
    cout << M << endl << endl;
    
    // Инвертируем матрицу как в нашем коде
    Mat M_inv = Mat::eye(2, 3, CV_64F);
    
    Mat A = M(cv::Rect(0, 0, 2, 2)).clone();
    Mat A_inv = A.inv();
    Mat t = M(cv::Rect(2, 0, 1, 2)).clone();
    Mat t_inv = -A_inv * t;
    
    A_inv.copyTo(M_inv(cv::Rect(0, 0, 2, 2)));
    t_inv.copyTo(M_inv(cv::Rect(2, 0, 1, 2)));
    
    cout << "Inverted matrix M_inv (target -> source):" << endl;
    cout << M_inv << endl << endl;
    
    // Теперь проверим несколько точек
    vector<Point2f> test_points = {
        Point2f(320, 240),  // центр (должен остаться центром)
        Point2f(320, 0),    // верх
        Point2f(640, 240),  // право
        Point2f(320, 480),  // низ
        Point2f(0, 240),    // лево
    };
    
    cout << fixed << setprecision(2);
    cout << "Point testing:" << endl;
    cout << "Target (dst) -> Source (src via M_inv)" << endl;
    cout << "---" << endl;
    
    for (auto& pt : test_points) {
        float src_x = M_inv.at<double>(0, 0) * pt.x + M_inv.at<double>(0, 1) * pt.y + M_inv.at<double>(0, 2);
        float src_y = M_inv.at<double>(1, 0) * pt.x + M_inv.at<double>(1, 1) * pt.y + M_inv.at<double>(1, 2);
        
        cout << "Target: (" << pt.x << ", " << pt.y << ") -> Source: (" 
             << src_x << ", " << src_y << ")" << endl;
    }
    
    cout << "\n---\n" << endl;
    cout << "Now let's verify using OpenCV's perspective transform:" << endl;
    
    // Используем cv::perspectiveTransform для проверки
    // Сначала преобразуем M в матрицу 3x3 для perspectiveTransform
    Mat M_3x3 = Mat::eye(3, 3, CV_64F);
    M.copyTo(M_3x3(cv::Rect(0, 0, 3, 2)));
    
    Mat M_3x3_inv = M_3x3.inv();
    
    cout << "M_3x3:" << endl << M_3x3 << endl;
    cout << "M_3x3_inv:" << endl << M_3x3_inv << endl << endl;
    
    Mat dst_pts(test_points);
    Mat src_pts;
    perspectiveTransform(dst_pts, src_pts, M_3x3_inv);
    
    cout << "perspectiveTransform result:" << endl;
    for (int i = 0; i < src_pts.rows; i++) {
        cout << "Target: (" << test_points[i].x << ", " << test_points[i].y << ") -> Source: ("
             << src_pts.at<float>(i, 0) << ", " << src_pts.at<float>(i, 1) << ")" << endl;
    }
    
    return 0;
}
