//
//#include <opencv2/core.hpp>
//#include <opencv2/videoio.hpp>
//#include <opencv2/highgui.hpp>
//
//#include "opencv2/imgcodecs.hpp"
//#include "opencv2/highgui.hpp"
//#include "opencv2/imgproc.hpp"
//
//#include <iostream>
//#include <stdio.h>
//
//#include "Config.hpp"
//#include "basicFunctions.hpp"
//#include "stabilizationFunctions.hpp"
//
//using namespace cv;
//using namespace std;
//
//#define NCoef 10
//#define DCgain 4
//
//#define Ntap 31
//
//TransformParam iirNoise(TransformParam &NewSample,vector<TransformParam>& x, vector<TransformParam>& y) {
//    
//    double FIRCoef[Ntap] = {
//          -40,
//          -16,
//           28,
//           48,
//           21,
//          -31,
//          -56,
//          -25,
//           33,
//           62,
//           29,
//          -34,
//          -66,
//          -32,
//           34,
//           68,
//           34,
//          -32,
//          -66,
//          -34,
//           29,
//           62,
//           33,
//          -25,
//          -56,
//          -31,
//           21,
//           48,
//           28,
//          -16,
//          -40
//    };
//    
//    double ACoef[NCoef+1] = {
//           12,
//            0,
//          -60,
//            0,
//          120,
//            0,
//         -120,
//            0,
//           60,
//            0,
//          -12
//    };
//
//    double BCoef[NCoef+1] = {
//           64,
//          -70,
//           30,
//          -16,
//           29,
//          -17,
//            5,
//           -1,
//            1,
//            0,
//            0
//    };
//
//    int n;
//
//    //shift the old samples
//    for(n=NCoef; n>0; n--) {
//       x[n] = x[n-1];
//       y[n] = y[n-1];
//    }
//
//    //Calculate the new output
//    x[0] = NewSample;
//    y[0].dx = ACoef[0] * x[0].dx;
//    y[0].dy = ACoef[0] * x[0].dy;
//    y[0].da = ACoef[0] * x[0].da;
//
//    for (n = 1; n <= NCoef; n++)
//    {
//        y[0].dx += ACoef[n] * x[n].dx - BCoef[n] * y[n].dx;
//        y[0].dy += ACoef[n] * x[n].dy - BCoef[n] * y[n].dy;
//        y[0].da += ACoef[n] * x[n].da - BCoef[n] * y[n].da;
//
//    }
//
//    y[0].dy /= (BCoef[0]*DCgain);
//    y[0].da /= (BCoef[0]*DCgain);
//    y[0].dx /= (BCoef[0]*DCgain);
//
//    return y[0];
//}
//
//
//int main(int, char**)
//{
//    Mat src, out, TStab(2, 3, CV_64F);
//    TransformParam noiseIn, noiseOut = { 0.0, 0.0, 0.0 };
//    vector <TransformParam> X(1+NCoef), Y(1 + NCoef);
//    RNG rng;
//
//    // use default camera as video source
//    VideoCapture cap(videoSource);
//    //VideoCapture cap(0);
//    // check if we succeeded
//    if (!cap.isOpened()) {
//        cerr << "ERROR! Unable to open camera\n";
//        return -1;
//    }
//    // get one frame from camera to know frame size and type
//    cap >> src;
//    // check if we succeeded
//    if (src.empty()) {
//        cerr << "ERROR! blank frame grabbed\n";
//        return -1;
//    }
//    bool isColor = (src.type() == CV_8UC3);
//    Rect roi;
//
//    roi.x = src.cols * 1 / 8;
//    roi.y = src.rows * 1 / 8;
//    roi.width = src.cols * 3 / 4;
//    roi.height = src.rows * 3 / 4;
//
//    //--- INITIALIZE VIDEOWRITER
//    VideoWriter writer;
//    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
//
//    double fps = 30.0;
//    string filename = "./OutputVideos/MoveLeftRoadShakedVideo.avi";
//
//    writer.open(filename, codec, fps, roi.size(), isColor);
//
//    // check if we succeeded
//    if (!writer.isOpened()) {
//        cerr << "Could not open the output video file for write\n";
//        return -1;
//    }
//
//    //--- GRAB AND WRITE LOOP
//    cout << "Writing videofile: " << filename << endl
//        << "Press any key to terminate" << endl;
//
//    //~~~~~~~~~~~~~~~~~~~~~~~~~~~Создаем объект для записи отклонения в CSV файл~~~~~~~~~~~~~~~~~~~~~~~~~~~
//    std::ofstream outputFile("./OutputResults/StabOutputs.txt");
//    if (!outputFile.is_open())
//    {
//        cout << "Не удалось открыть файл для записи" << endl;
//        return -1;
//    }
//
//
//    for (;;)
//    {
//        cap >> src;
//        // check if we succeeded
//        if (src.empty()) {
//            cerr << "Ending.\n";
//            break;
//        }
//        noiseIn.dx = (double)(rng.uniform(-100.0, 100.0)) /32          ;// / 32 + noiseIn.dx * 31 / 32;
//        noiseIn.dy = (double)(rng.uniform(-100.0, 100.0)) /32          ;//    / 32 + noiseIn.dy * 31 / 32;
//        noiseIn.da = (double)(rng.uniform(-1000.0, 1000.0) * 0.0001)/32;// / 32 + noiseIn.da * 31 / 32;
//
//        noiseOut = iirNoise(noiseIn, X,Y);
//
//        noiseOut.getTransform(TStab);
//        warpAffine(src, out, TStab, src.size());
//
//        out = out(roi);
//        // encode the frame into the videofile stream
//        writer.write(out);
//        // show live and wait for a key with timeout long enough to show images
//        imshow("Live", out);
//        if (waitKey(1) >= 0)
//            break;
//    }
//    // the videofile will be closed and released automatically in VideoWriter destructor
//    return 0;
//}
////
////#include "Config.hpp"
////#include "basicFunctions.hpp"
////#include "stabilizationFunctions.hpp"
////
////int main() {
////    // System dimensions
////    int state_dim = 4;  // vx, vy, ax, ay
////    int meas_dim = 2;   // vx, vy
////
////    // Create system matrices
////    double FPS = 30.0;
////    double dt = 1;//1 / FPS;
////    cv::Mat A = (cv::Mat_<double>(4, 4) <<
////        1, 0, dt, 0,
////        0, 1, 0, dt,
////        0, 0, 1, 0,
////        0, 0, 0, 1);
////
////    cv::Mat C = (cv::Mat_<double>(2, 4) <<
////        1, 0, 0, 0,
////        0, 1, 0, 0);
////
////    cv::Mat Q = cv::Mat::eye(4, 4, CV_64F) * 0.01;
////    cv::Mat R = cv::Mat::eye(2, 2, CV_64F) * 0.1;
////    cv::Mat P = cv::Mat::eye(4, 4, CV_64F) * 1;
////
////    // Create Kalman filter
////    KalmanFilterCV kf(dt, A, C, Q, R, P);
////
////    // Initialize with first measurement
////    cv::Mat x0 = (cv::Mat_<double>(4, 1) << 0, 0, 0, 0);
////    kf.init(0, x0);
////
////    // Simulate measurements and update
////    for (int i = 0; i < 40; i++) {
////        cv::Mat measurement = (cv::Mat_<double>(2, 1) << i * 1.0, i * 0.5);
////        std::cout << "measurement: " << measurement.t() << std::endl;
////        kf.update(measurement);
////
////        cv::Mat state = kf.state();
////        std::cout << "State: " << state.t() << std::endl;
////    }
////
////    return 0;
////}