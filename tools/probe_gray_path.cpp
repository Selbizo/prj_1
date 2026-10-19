// Probe 6: J — INTER_LINEAR по 2-канальному YUYV, затем YUYV2GRAY на мелком.
// Канал 0 у YUYV везде равен Y, поэтому линейная интерполяция канала 0
// эквивалентна линейному уменьшению Y-плоскости — качество как у эталона F.
//  E: resize YUYV NEAREST -> GRAY   (быстро, алиасинг)
//  F: GRAY полный -> resize LINEAR  (эталон качества, текущий код)
//  J: resize YUYV LINEAR -> GRAY    (кандидат: качество F, скорость E)
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
using namespace cv; using namespace std;
using clk = chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b){
    return chrono::duration_cast<chrono::nanoseconds>(b-a).count()/1e6;
}
static VideoCapture openRaw(){
    VideoCapture c; c.open("/dev/video0", CAP_V4L2);
    c.set(CAP_PROP_FRAME_WIDTH,1920); c.set(CAP_PROP_FRAME_HEIGHT,1080);
    c.set(CAP_PROP_FOURCC,VideoWriter::fourcc('Y','U','Y','V'));
    c.set(CAP_PROP_CONVERT_RGB,0);
    return c;
}
static double median(vector<double> v){ sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(){
    setbuf(stdout,NULL);
    const Size track(384,216);
    Mat raw;
    const int ROUNDS=5, N=60;
    vector<double> eT,fT,jT;

    VideoCapture c=openRaw();
    if(!c.isOpened()){ printf("cannot open\n"); return 1; }
    for(int i=0;i<40;i++) c.read(raw);

    { // корректность относительно эталона F
        c.read(raw); Mat y=raw.reshape(2,1080);
        Mat fullGray; cvtColor(y,fullGray,COLOR_YUV2GRAY_YUY2);
        Mat ref;      resize(fullGray,ref,track,0,0,INTER_LINEAR);
        Mat j2; resize(y,j2,track,0,0,INTER_LINEAR);
        Mat j;  cvtColor(j2,j,COLOR_YUV2GRAY_YUY2);
        Mat e;  resize(y,e,track,0,0,INTER_NEAREST); cvtColor(e,e,COLOR_YUV2GRAY_YUY2);
        Mat dJ,dE; absdiff(ref,j,dJ); absdiff(ref,e,dE);
        double xJ,xE; Scalar mJ=mean(dJ), mE=mean(dE);
        minMaxLoc(dJ,nullptr,&xJ); minMaxLoc(dE,nullptr,&xE);
        printf("=== корректность (отличие от эталона F) ===\n");
        printf("  J (YUYV LINEAR)  : среднее=%.3f максимум=%.0f\n", mJ[0], xJ);
        printf("  E (YUYV NEAREST) : среднее=%.2f максимум=%.0f\n", mE[0], xE);
        imwrite("/tmp/opencode/J.png",j); imwrite("/tmp/opencode/F.png",ref);
    }

    for(int r=0;r<ROUNDS;r++){
        { for(int i=0;i<N;i++){ auto a=clk::now(); c.read(raw);
            Mat y=raw.reshape(2,1080); resize(y,y,track,0,0,INTER_NEAREST);
            cvtColor(y,y,COLOR_YUV2GRAY_YUY2); auto z=clk::now(); eT.push_back(ms(a,z)); } }
        { for(int i=0;i<N;i++){ auto a=clk::now(); c.read(raw);
            Mat y=raw.reshape(2,1080); Mat g; cvtColor(y,g,COLOR_YUV2GRAY_YUY2);
            resize(g,y,track,0,0,INTER_LINEAR); auto z=clk::now(); fT.push_back(ms(a,z)); } }
        { for(int i=0;i<N;i++){ auto a=clk::now(); c.read(raw);
            Mat y=raw.reshape(2,1080); resize(y,y,track,0,0,INTER_LINEAR);
            cvtColor(y,y,COLOR_YUV2GRAY_YUY2); auto z=clk::now(); jT.push_back(ms(a,z)); } }
        printf("раунд %d: E=%.2f  F=%.2f  J=%.2f мс\n", r, eT.back(), fT.back(), jT.back());
    }
    printf("\n=== медианы ===\n");
    printf("  E (NEAREST)      : %.2f мс = %.1f fps\n", median(eT),1000.0/median(eT));
    printf("  F (эталон)       : %.2f мс = %.1f fps\n", median(fT),1000.0/median(fT));
    printf("  J (LINEAR YUYV)  : %.2f мс = %.1f fps\n", median(jT),1000.0/median(jT));
    printf("\n  интервал кадра сенсора = 8.53 мс (117 fps)\n");
    return 0;
}
