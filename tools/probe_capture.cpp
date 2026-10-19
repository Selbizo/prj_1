// Probe: где в конвейере выживает 120 fps.
// Замеряет путь захвата и стоимость оценки смещения (детекция + трекинг)
// на реальном /dev/video0, тем же способом, что и video_stabilizer.cpp.
#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdio>
using namespace cv; using namespace std;
using clk = chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b){
    return chrono::duration_cast<chrono::nanoseconds>(b-a).count()/1e6;
}
static VideoCapture openCam(bool convertRGB){
    VideoCapture c;
    c.open("/dev/video0", CAP_V4L2);
    if(!c.isOpened()) return c;
    c.set(CAP_PROP_FRAME_WIDTH,1920);
    c.set(CAP_PROP_FRAME_HEIGHT,1080);
    c.set(CAP_PROP_FOURCC,VideoWriter::fourcc('Y','U','Y','V'));
    if(!convertRGB) c.set(CAP_PROP_CONVERT_RGB,0);
    return c;
}
// raw V4L2 без конверсии приходит как Nx1 8UC1 -> приводим к 1080x1920 2ch
static Mat unflatten(const Mat& raw,int rows,int cols){
    if(CV_MAT_CN(raw.type())==2) return raw;
    if(raw.total()==(size_t)rows*cols*2) return raw.reshape(2,rows);
    return Mat();
}
int main(){
    setbuf(stdout,NULL);
    const Size track(384,216);        // 1920/5 x 1080/5 (compressionConfig=5)

    // ---------- 1. скорость сенсора: чтение сырых байт без конверсии ----------
    {
        VideoCapture c = openCam(false);
        if(!c.isOpened()){ printf("cannot open /dev/video0\n"); return 1; }
        Mat raw; for(int i=0;i<20;i++) c.read(raw);
        Mat f = unflatten(raw,1080,1920);
        printf("=== 1. Сенсор: чтение raw без конверсии ===\n");
        printf("  raw=%dx%d type=%d ch=%d -> %dx%d ch=%d\n", raw.cols,raw.rows,raw.type(),
               CV_MAT_CN(raw.type()), f.cols,f.rows,CV_MAT_CN(f.type()));
        auto t0=clk::now(); int n=0;
        while(ms(t0,clk::now())<3000.0){ if(c.read(raw)) n++; }
        double dt=ms(t0,clk::now());
        printf("  %d кадров за %.0f мс = %.1f fps  (%.2f мс/кадр)\n\n", n, dt, n*1000.0/dt, dt/n);
        c.release();
    }

    Mat gray, small;
    // ---------- 2. Путь A: как в приложении сейчас (BGR -> GRAY -> resize AREA) ----------
    {
        VideoCapture c = openCam(true); Mat bgr; for(int i=0;i<20;i++) c.read(bgr);
        double tR=0,tC=0,tS=0; const int N=120;
        for(int i=0;i<N;i++){
            auto a=clk::now(); c.read(bgr);                    auto b=clk::now();
            cvtColor(bgr,gray,COLOR_BGR2GRAY);                 auto d=clk::now();
            resize(gray,small,track,0,0,INTER_AREA);           auto e=clk::now();
            tR+=ms(a,b); tC+=ms(b,d); tS+=ms(d,e);
        }
        double t=(tR+tC+tS)/N;
        printf("=== 2. Захват A: read(BGR) + BGR2GRAY + resize INTER_AREA ===\n");
        printf("  read=%.2f  gray=%.2f  resize=%.2f  ->  %.2f мс = %.1f fps\n\n",tR/N,tC/N,tS/N,t,1000.0/t);
        c.release();
    }
    // ---------- 3. Путь B: VS_FAST_RESIZE=1 (LINEAR) ----------
    {
        VideoCapture c = openCam(true); Mat bgr; for(int i=0;i<20;i++) c.read(bgr);
        double tR=0,tC=0,tS=0; const int N=120;
        for(int i=0;i<N;i++){
            auto a=clk::now(); c.read(bgr);                    auto b=clk::now();
            cvtColor(bgr,gray,COLOR_BGR2GRAY);                 auto d=clk::now();
            resize(gray,small,track,0,0,INTER_LINEAR);         auto e=clk::now();
            tR+=ms(a,b); tC+=ms(b,d); tS+=ms(d,e);
        }
        double t=(tR+tC+tS)/N;
        printf("=== 3. Захват B: read(BGR) + BGR2GRAY + resize LINEAR ===\n");
        printf("  read=%.2f  gray=%.2f  resize=%.2f  ->  %.2f мс = %.1f fps\n\n",tR/N,tC/N,tS/N,t,1000.0/t);
        c.release();
    }
    // ---------- 4. Путь C: raw YUYV -> GRAY напрямую -> resize ----------
    {
        VideoCapture c = openCam(false); Mat raw; for(int i=0;i<20;i++) c.read(raw);
        Mat y = unflatten(raw,1080,1920);
        if(y.empty()){ printf("=== 4. Путь C: не удалось распаковать raw YUYV ===\n\n"); }
        else {
            double tR=0,tC=0,tS=0; const int N=120;
            for(int i=0;i<N;i++){
                auto a=clk::now(); c.read(raw);                    auto b=clk::now();
                Mat yf = unflatten(raw,1080,1920);
                cvtColor(yf,gray,COLOR_YUV2GRAY_YUY2);             auto d=clk::now();
                resize(gray,small,track,0,0,INTER_LINEAR);         auto e=clk::now();
                tR+=ms(a,b); tC+=ms(b,d); tS+=ms(d,e);
            }
            double t=(tR+tC+tS)/N;
            printf("=== 4. Захват C: read(raw YUYV) + YUYV2GRAY + resize LINEAR ===\n");
            printf("  read=%.2f  gray=%.2f  resize=%.2f  ->  %.2f мс = %.1f fps\n\n",tR/N,tC/N,tS/N,t,1000.0/t);
        }
        c.release();
    }
    // ---------- 5. Оценка смещения: GFTT + LK на 384x216 ----------
    {
        VideoCapture c = openCam(true); Mat bgr; for(int i=0;i<40;i++) c.read(bgr);
        cvtColor(bgr,gray,COLOR_BGR2GRAY); resize(gray,small,track,0,0,INTER_LINEAR);
        Ptr<GFTTDetector> det = GFTTDetector::create(200, 0.005, 7.0, 9, true, 0.005);
        // маска как в приложении: поле 3/4 кадра по центру
        Mat maskCPU = Mat::zeros(small.size(), CV_8U);
        int mx = small.cols/8, my = small.rows/8;
        rectangle(maskCPU, Rect(mx,my,small.cols-2*mx,small.rows-2*my), Scalar(255), FILLED);
        vector<KeyPoint> kp; det->detect(small,kp,maskCPU);
        vector<Point2f> prev,next; KeyPoint::convert(kp,prev);
        Mat prevGray = small.clone();
        vector<uchar> st; vector<float> err;
        const int win=9,maxLev=5; auto crit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,10,0.03);
        // детекция (полная передетекция — в приложении раз в 15 кадров)
        double tD=0; const int N=60;
        for(int i=0;i<N;i++){
            c.read(bgr); cvtColor(bgr,gray,COLOR_BGR2GRAY); resize(gray,small,track,0,0,INTER_LINEAR);
            vector<KeyPoint> p; auto a=clk::now(); det->detect(small,p,maskCPU); auto b=clk::now(); tD+=ms(a,b);
        }
        // трекинг LK
        double tT=0;
        for(int i=0;i<N;i++){
            c.read(bgr); cvtColor(bgr,gray,COLOR_BGR2GRAY); resize(gray,small,track,0,0,INTER_LINEAR);
            auto a=clk::now();
            calcOpticalFlowPyrLK(prevGray, small, prev, next, st, err, Size(win,win), maxLev, crit);
            auto b=clk::now(); tT+=ms(a,b); small.copyTo(prevGray);
        }
        printf("=== 5. Оценка смещения на %dx%d (как в приложении) ===\n", track.width, track.height);
        printf("  GFTT детекция (200 точек, маска) = %.2f мс\n", tD/N);
        printf("  LK трекинг (win=%d lvl=%d)        = %.2f мс\n", win, maxLev, tT/N);
        printf("  среднее при передетекции раз в 15 кадров = %.2f мс = %.1f fps\n",
               tT/N + tD/N/15.0, 1000.0/(tT/N + tD/N/15.0));
        printf("  худший случай (детекция каждый кадр)     = %.2f мс = %.1f fps\n",
               tD/N+tT/N, 1000.0/(tD/N+tT/N));
        c.release();
    }
    // ---------- 6. оценка аффинного преобразования (RANSAC) ----------
    {
        vector<Point2f> a,b;
        RNG rng(12345);
        for(int i=0;i<180;i++){ Point2f p(rng.uniform(20.f,360.f),rng.uniform(20.f,200.f)); a.push_back(p);
                                b.push_back(Point2f(p.x+0.7f+rng.uniform(-0.3f,0.3f), p.y-0.4f+rng.uniform(-0.3f,0.3f))); }
        double t=0; const int N=200;
        for(int i=0;i<N;i++){ auto s=clk::now(); Mat T=estimateAffine2D(a,b,noArray(),RANSAC,3.0); auto e=clk::now(); t+=ms(s,e); }
        printf("=== 6. estimateAffine2D RANSAC (180 точек) = %.2f мс\n", t/N);
        printf("    -> итоговая оценка смещения = LK 1.10 + %.2f = %.2f мс (передетекция раз в 15 кадров)\n\n",
               t/N, 1.10 + t/N + 0.40);
    }
    printf("\nБюджет 120 fps = 8.33 мс на кадр.\n");
    return 0;
}
