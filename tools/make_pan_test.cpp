// Генератор тестового видео с ИЗВЕСТНЫМ смещением: контент панорамируется
// на +2.0 px вправо и +1.0 px вниз за кадр (полное разрешение 1920x1080).
// Панорама делается вырезкой окна из большого полотна, поэтому нет артефактов
// на краях, а смещение между соседними кадрами строго постоянно.
#include <opencv2/opencv.hpp>
#include <cstdio>
using namespace cv; using namespace std;

int main(){
    const int W=1920, H=1080, N=120;
    const double DX=2.0, DY=1.0;                 // px/кадр, полное разрешение
    const int CW = W + (int)(DX*(N-1)) + 4;      // полотно с запасом
    const int CH = H + (int)(DY*(N-1)) + 4;

    Mat canvas(CH, CW, CV_8UC1, Scalar(28));
    RNG rng(42);
    for(int i=0;i<420;i++){
        Point c(rng.uniform(0,CW-1), rng.uniform(0,CH-1));
        int r = rng.uniform(6,26);
        int v = rng.uniform(90,255);
        circle(canvas, c, r, Scalar(v), FILLED, LINE_AA);
    } // end for (i) — случайные круги = устойчивые углы для GFTT

    VideoWriter w("/tmp/opencode/pan.mp4",
                  VideoWriter::fourcc('m','p','4','v'), 30.0, Size(W,H));
    if(!w.isOpened()){ printf("cannot open writer\n"); return 1; }
    for(int f=0; f<N; f++){
        Rect r((int)(DX*f), (int)(DY*f), W, H);
        Mat frame = canvas(r);
        Mat frame3; cvtColor(frame, frame3, COLOR_GRAY2BGR);  // mp4v требует 3 канала
        w.write(frame3);
    } // end for (f)
    w.release();
    printf("Готово: /tmp/opencode/pan.mp4  %dx%d, %d кадров, смещение +%.1f/+%.1f px за кадр\n",
           W,H,N,DX,DY);
    printf("Ожидаемое в [EST]: dx=+2.00 dy=+1.00 da=0.000\n");
    return 0;
}
