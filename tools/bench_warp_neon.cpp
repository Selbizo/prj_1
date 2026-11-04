// Бенчмарк: cv::warpAffine (CPU) vs NEON-ядро.
// Сборка:
//   g++ -O3 -std=c++17 -I../src \
//       $(pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/local/opencv-4.12.0/include/opencv4) \
//       bench_warp_neon.cpp ../src/gpu/warp_affine_neon.cpp \
//       $(pkg-config --libs opencv4 2>/dev/null || echo -L/usr/local/opencv-4.12.0/lib -lopencv_core -lopencv_imgproc) \
//       -o /tmp/bench_warp_neon

#include "gpu/warp_affine_neon.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace cv;
using namespace std;

static double median(vector<double> v) {
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static Mat makeM(double angleDeg, double tx, double ty, double scale = 1.0) {
    const double a = angleDeg * CV_PI / 180.0;
    Mat M = (Mat_<double>(2, 3) <<
             scale * cos(a), -scale * sin(a), tx,
             scale * sin(a),  scale * cos(a), ty);
    return M;
}

struct Stat {
    double maxAbs = 0;
    double meanAbs = 0;
    double bad = 0;  // доля пикселей с |diff| > 1
};

static Stat compare(const Mat& a, const Mat& b) {
    Stat s;
    Mat d;
    absdiff(a, b, d);
    double total = 0;
    double bad = 0;
    for (int y = 0; y < d.rows; ++y) {
        const uchar* p = d.ptr(y);
        for (int x = 0; x < d.cols * d.channels(); ++x) {
            const int v = p[x];
            total += v;
            if (v > 1) bad += 1;
            s.maxAbs = max(s.maxAbs, (double)v);
        }
    }
    const double n = (double)d.total() * d.channels();
    s.meanAbs = total / n;
    s.bad = bad / n;
    return s;
}

int main() {
    printf("NEON available: %s\n", warpAffineNEONAvailable() ? "yes" : "no");

    const Size srcSize(1920, 1080);
    const Size dstSize(1344, 756);

    // Похоже на реальный кадр: плавные градиенты + немного шума и резких краёв.
    Mat src(srcSize, CV_8UC3);
    RNG rng(12345);
    for (int y = 0; y < src.rows; ++y) {
        Vec3b* p = src.ptr<Vec3b>(y);
        for (int x = 0; x < src.cols; ++x) {
            p[x][0] = (uchar)((x * 255 / src.cols + y * 3) & 255);
            p[x][1] = (uchar)((y * 255 / src.rows + x * 2) & 255);
            p[x][2] = (uchar)(((x ^ y) * 7) & 255);
        }
    }
    for (int i = 0; i < 20000; ++i) {
        int x = rng.uniform(0, src.cols), y = rng.uniform(0, src.rows);
        src.at<Vec3b>(y, x) = Vec3b(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    struct Case { const char* name; double angle, tx, ty, scale; };
    const vector<Case> cases = {
        {"identity",              0.0,   0.0,   0.0, 1.0},
        {"rot +0.5deg",           0.5,  20.0, -15.0, 1.0},
        {"rot -1.0deg",          -1.0, -30.0,  25.0, 1.0},
        {"rot +2.0deg",           2.0,  40.0, -35.0, 1.0},
        {"zoom 1.03 + rot 0.7",   0.7,  10.0,  10.0, 1.03},
        {"big shift (border)",    0.3, 120.0,-110.0, 1.0},
    };

    const int reps = 9;
    const int warmup = 3;

    printf("\n%-22s | %-11s | %-11s | %-11s | %-11s | %s\n",
           "case", "cv 1thr ms", "NEON 1thr", "cv 6thr", "NEON 4thr", "diff max/mean/>1");
    printf("--------------------------------------------------------------------------------------------------\n");

    for (const auto& c : cases) {
        Mat M = makeM(c.angle, c.tx, c.ty, c.scale);
        Mat ref, out;

        // OpenCV single-thread
        cv::setNumThreads(1);
        for (int i = 0; i < warmup; ++i)
            warpAffine(src, ref, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        vector<double> tv;
        for (int i = 0; i < reps; ++i) {
            int64 t0 = getTickCount();
            warpAffine(src, ref, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
            tv.push_back((getTickCount() - t0) * 1000.0 / getTickFrequency());
        }
        const double cv1 = median(tv);

        // NEON single-thread
        for (int i = 0; i < warmup; ++i)
            warpAffineNEON(src, out, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), 1);
        tv.clear();
        for (int i = 0; i < reps; ++i) {
            int64 t0 = getTickCount();
            warpAffineNEON(src, out, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), 1);
            tv.push_back((getTickCount() - t0) * 1000.0 / getTickFrequency());
        }
        const double neon1 = median(tv);
        const Stat st = compare(ref, out);

        // OpenCV default pool (6 cores)
        cv::setNumThreads(6);
        for (int i = 0; i < warmup; ++i)
            warpAffine(src, ref, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        tv.clear();
        for (int i = 0; i < reps; ++i) {
            int64 t0 = getTickCount();
            warpAffine(src, ref, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
            tv.push_back((getTickCount() - t0) * 1000.0 / getTickFrequency());
        }
        const double cv6 = median(tv);

        // NEON 4 threads
        for (int i = 0; i < warmup; ++i)
            warpAffineNEON(src, out, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), 4);
        tv.clear();
        for (int i = 0; i < reps; ++i) {
            int64 t0 = getTickCount();
            warpAffineNEON(src, out, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), 4);
            tv.push_back((getTickCount() - t0) * 1000.0 / getTickFrequency());
        }
        const double neon4 = median(tv);
        cv::setNumThreads(1);

        printf("%-22s | %11.3f | %11.3f | %11.3f | %11.3f | %.0f / %.4f / %.3f%%\n",
               c.name, cv1, neon1, cv6, neon4,
               st.maxAbs, st.meanAbs, st.bad * 100.0);
    }

    // Более «жёсткая» проверка соответствия: много случайных матриц.
    double worstMax = 0, worstMean = 0;
    cv::setNumThreads(1);
    for (int t = 0; t < 50; ++t) {
        Mat M = makeM(rng.uniform(-3.0, 3.0), rng.uniform(-60.0, 60.0),
                      rng.uniform(-60.0, 60.0), rng.uniform(0.95, 1.05));
        Mat a, b;
        warpAffine(src, a, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar());
        warpAffineNEON(src, b, M, dstSize, INTER_LINEAR, BORDER_CONSTANT, Scalar(), 1);
        Stat s = compare(a, b);
        worstMax = max(worstMax, s.maxAbs);
        worstMean = max(worstMean, s.meanAbs);
    }
    printf("\n50 random affines: worst max|diff| = %.0f, worst mean|diff| = %.4f\n",
           worstMax, worstMean);
    return 0;
}
