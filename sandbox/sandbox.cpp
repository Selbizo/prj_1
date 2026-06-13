
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

#include "Config.hpp"
#include "basicFunctions.hpp"
#include "stabilizationFunctions.hpp"
#include "wienerFilter.hpp"

#include <random>



using namespace cv;
using namespace std;

#define NCoef 10
#define DCgain 4

#define Ntap 31

void addGaussianNoise(cv::Mat &image, double mean = 0, double stddev = 20) {
    cv::Mat noise(image.size(), image.type());
    cv::randn(noise, mean, stddev); // Генерация шума
    image += noise; // Добавление шума к изображению
}
TransformParam iirNoise(TransformParam &NewSample,vector<TransformParam>& x, vector<TransformParam>& y) {
   
   double FIRCoef[Ntap] = {
         -40, -16, 28, 48, 21, -31, -56, -25, 33, 62, 29, -34, -66, -32, 34, 68, 34, -32, -66, -34, 29, 62, 33, -25, -56,-31, 21, 48, 28, -16, -40
   };
   
   double ACoef[NCoef+1] = {
          12, 0, -60, 0, 120, 0, -120, 0, 60, 0, -12
   };

   double BCoef[NCoef+1] = {
          64, -70, 30, -16, 29, -17, 5, -1, 1, 0, 0
   };

   int n;

   //shift the old samples
   for(n=NCoef; n>0; n--) {
      x[n] = x[n-1];
      y[n] = y[n-1];
   }

   //Calculate the new output
   x[0] = NewSample;
   y[0].dx = ACoef[0] * x[0].dx;
   y[0].dy = ACoef[0] * x[0].dy;
   y[0].da = ACoef[0] * x[0].da;

   for (n = 1; n <= NCoef; n++)
   {
       y[0].dx += ACoef[n] * x[n].dx - BCoef[n] * y[n].dx;
       y[0].dy += ACoef[n] * x[n].dy - BCoef[n] * y[n].dy;
       y[0].da += ACoef[n] * x[n].da - BCoef[n] * y[n].da;

   }

   y[0].dy /= (BCoef[0]*DCgain);
   y[0].da /= (BCoef[0]*DCgain);
   y[0].dx /= (BCoef[0]*DCgain);

   return y[0];
}


int main(int, char**)
{
   Mat src, out, TStab(2, 3, CV_64F);

   TransformParam noiseIn = { 0.0, 0.0, 0.0 };
   vector <TransformParam> noiseOut(2);
	for (int i = 0; i < noiseOut.size();i++)
	{
		noiseOut[i].dx = 0.0;
		noiseOut[i].dy = 0.0;
		noiseOut[i].da = 0.0;
	}

   vector <TransformParam> X(1+NCoef), Y(1 + NCoef);
   double LEN, THETA;
   RNG rng;
   
   // use default camera as video source
   VideoCapture cap(videoSourceForShaked);
   //VideoCapture cap(0);
   // check if we succeeded
   if (!cap.isOpened()) {
       cerr << "ERROR! Unable to open camera\n";
       return -1;
    }
    // get one frame from camera to know frame size and type
    cap >> src;
    // check if we succeeded
    if (src.empty()) {
        cerr << "ERROR! blank frame grabbed\n";
        return -1;
    }
    cv::Mat Smooth(Size(src.cols, src.rows), CV_32F, cv::Scalar(0.0,0.0,0.0));

    bool isColor = (src.type() == CV_8UC3);
    Rect roi;
    
    roi.x = src.cols * 1 / 16;
    roi.y = src.rows * 1 / 16;
    roi.width = src.cols * 7 / 8;
    roi.height = src.rows * 7 / 8;
    
   //--- INITIALIZE VIDEOWRITER
   VideoWriter writer;
   int codec = VideoWriter::fourcc('a', 'v', 'c', '1');
   double fps = 30.0;
   string filename = "./SourceVideos/FlightShakedVideo.mp4";

   writer.open(filename, codec, fps, roi.size(), isColor);

   // check if we succeeded
   if (!writer.isOpened()) {
       cerr << "Could not open the output video file for write\n";
       return -1;
   }

   //--- GRAB AND WRITE LOOP
   cout << "Writing videofile: " << filename << endl
       << "Press any key to terminate" << endl;

   std::ofstream outputFile("./OutputResults/StabOutputs.txt");
   if (!outputFile.is_open())
   {
       cout << "Error open text file " << endl;
       return -1;
   }


   short cnt = 0;
   for (;;)
   {
       cap >> src;
       // check if we succeeded
       if (src.empty()) {
           cerr << "Ending.\n";
           break;
       }
       noiseIn.dx = (double)(rng.uniform(-100.0, 100.0)) /4          ;// / 32 + noiseIn.dx * 31 / 32;
       noiseIn.dy = (double)(rng.uniform(-100.0, 100.0)) /4          ;//    / 32 + noiseIn.dy * 31 / 32;
       noiseIn.da = (double)(rng.uniform(-1000.0, 1000.0) * 0.0001)/8;// / 32 + noiseIn.da * 31 / 32;

       noiseOut[0] = iirNoise(noiseIn, X,Y);

       noiseOut[0].getTransform(TStab);
       warpAffine(src, out, TStab, src.size());

        //добавить смаз

        LEN = sqrt((noiseOut[1].dx - noiseOut[0].dx)*(noiseOut[1].dx - noiseOut[0].dx) + (noiseOut[1].dy - noiseOut[0].dy)*(noiseOut[1].dy - noiseOut[0].dy))*0.7;
        if ((noiseOut[1].dx - noiseOut[0].dx) == 0.0)
            if ((noiseOut[1].dy - noiseOut[0].dy) > 0.0)
                THETA = 90.0;
            else
                THETA = -90.0;
        else
            THETA = atan((noiseOut[1].dy - noiseOut[0].dy) / (noiseOut[1].dx - noiseOut[0].dx)) * RAD_TO_DEG;

        calcPSF(Smooth, cv::Size((int)LEN * 1 + 10, (int)LEN * 1 + 10), LEN, THETA);
        cv::filter2D(out, out, -1, Smooth, cv::Point(-1,-1), 0, cv::BORDER_DEFAULT);

       out = out(roi);
       addGaussianNoise(out, 0, 10);

       // encode the frame into the videofile stream
       writer.write(out);
       // show live and wait for a key with timeout long enough to show images
       cnt++;
       if (cnt % 16 == 0)
            imshow("Live", out);

    noiseOut[1] = noiseOut[0];
       if (waitKey(1) >= 0)
           break;
   }
   // the videofile will be closed and released automatically in VideoWriter destructor
   return 0;
}