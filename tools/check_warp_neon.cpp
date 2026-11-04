// Проверка NEON-ядра против cv::warpAffine (bit-exact для поддержанных случаев).
// Сборка:
//   g++ -O2 -std=c++17 -I../src bench/check_warp_neon.cpp ../src/gpu/warp_affine_neon.cpp \
//       $(pkg-config --cflags --libs opencv4 2>/dev/null || \
//         echo -I/usr/local/opencv-4.12.0/include/opencv4 -L/usr/local/opencv-4.12.0/lib -lopencv_core -lopencv_imgproc) \
//       -o /tmp/check_warp_neon

#include "gpu/warp_affine_neon.h"

#include <opencv2/opencv.hpp>
#include <cmath>
#include <cstdio>
#include <string>

using namespace cv;
using namespace std;

static int gFailures = 0;
static int gChecks = 0;

static void check(const char* name, const Mat& a, const Mat& b) {
    ++gChecks;
    if (a.size() != b.size() || a.type() != b.type()) {
        printf("FAIL %-42s: size/type %dx%d/%d vs %dx%d/%d\n", name,
               a.cols, a.rows, a.type(), b.cols, b.rows, b.type());
        ++gFailures;
        return;
    }
    const double n = norm(a, b, NORM_INF);
    if (n != 0.0) {
        printf("FAIL %-42s: max|diff| = %.0f\n", name, n);
        ++gFailures;
    }
}

static Mat makeM(double angleDeg, double tx, double ty, double scale) {
    const double a = angleDeg * CV_PI / 180.0;
    return (Mat_<double>(2, 3) <<
            scale * cos(a), -scale * sin(a), tx,
            scale * sin(a),  scale * cos(a), ty);
}

int main() {
    if (!warpAffineNEONAvailable()) {
        printf("NEON недоступен — нечего проверять\n");
        return 77;
    }
    cv::setNumThreads(4);

    RNG rng(777);
    const int channels[] = {1, 3};
    const Size sizes[] = {Size(1, 1), Size(2, 2), Size(3, 2), Size(7, 5),
                          Size(17, 13), Size(64, 48), Size(101, 79)};
    const Size dsizeList[] = {Size(1, 1), Size(3, 2), Size(7, 5),
                              Size(17, 13), Size(64, 48), Size(101, 79)};
    const int borders[] = {0, 7, 255};
    const int threadList[] = {1, 2};

    int caseId = 0;
    for (int cn : channels) {
        for (Size ssz : sizes) {
            Mat src(ssz, CV_MAKE_TYPE(CV_8U, cn));
            rng.fill(src, RNG::UNIFORM, 0, 256);
            // Непрерывный и «подрезанный» (неcontinuous) источник.
            Mat roiSrc = src(Rect(0, 0, max(1, ssz.width - 1), ssz.height));
            const Mat* sources[] = {&src, &roiSrc};
            for (const Mat* sp : sources) {
                const Mat& s = *sp;
                for (Size dsz : dsizeList) {
                    if (dsz.width > s.cols + 4 || dsz.height > s.rows + 4) continue;
                    for (int borderValue : borders) {
                        for (int th : threadList) {
                            Mat M;
                            switch (caseId % 6) {
                                case 0: M = makeM(0.0, 0.0, 0.0, 1.0); break;
                                case 1: M = makeM(0.7, 1.5, -0.5, 1.0); break;
                                case 2: M = makeM(-2.0, -3.0, 2.0, 1.0); break;
                                case 3: M = makeM(4.0, 5.0, -4.0, 1.03); break;
                                case 4: M = makeM(0.0, 100.0, -100.0, 1.0); break;
                                default: M = makeM(0.5, -100.0, 100.0, 0.97); break;
                            }
                            for (int inverse = 0; inverse < 2; ++inverse) {
                                int flags = INTER_LINEAR;
                                Mat Mm = M;
                                if (inverse) {
                                    flags |= WARP_INVERSE_MAP;
                                    invertAffineTransform(M, Mm);
                                }
                                Mat ref, got;
                                warpAffine(s, ref, Mm, dsz, flags,
                                           BORDER_CONSTANT, Scalar::all(borderValue));
                                const bool ok = warpAffineNEON(
                                    s, got, Mm, dsz, flags,
                                    BORDER_CONSTANT, Scalar::all(borderValue), th);
                                ++gChecks;
                                if (!ok) {
                                    printf("FAIL case %d: NEON вернул false (cn=%d)\n",
                                           caseId, cn);
                                    ++gFailures;
                                    continue;
                                }
                                char name[192];
                                snprintf(name, sizeof(name),
                                         "cn=%d %dx%d -> %dx%d bv=%d th=%d inv=%d M=[%.3f %.3f %.3f; %.3f %.3f %.3f]",
                                         cn, s.cols, s.rows, dsz.width, dsz.height,
                                         borderValue, th, inverse,
                                         Mm.at<double>(0,0), Mm.at<double>(0,1), Mm.at<double>(0,2),
                                         Mm.at<double>(1,0), Mm.at<double>(1,1), Mm.at<double>(1,2));
                                check(name, ref, got);
                            }
                            ++caseId;
                        }
                    }
                }
            }
        }
    }

    // INTER_AREA в warpAffine — это INTER_LINEAR.
    {
        Mat src(41, 53, CV_8UC3);
        rng.fill(src, RNG::UNIFORM, 0, 256);
        Mat M = makeM(1.0, 2.0, -3.0, 1.0);
        Mat ref, got;
        warpAffine(src, ref, M, Size(37, 29), INTER_AREA, BORDER_CONSTANT, Scalar());
        const bool ok = warpAffineNEON(src, got, M, Size(37, 29), INTER_AREA,
                                       BORDER_CONSTANT, Scalar(), 1);
        ++gChecks;
        if (!ok) { printf("FAIL INTER_AREA: NEON вернул false\n"); ++gFailures; }
        else check("INTER_AREA", ref, got);
    }

    // Неподдержанные случаи должны возвращать false (вызывающий откатится на OpenCV).
    {
        Mat src(20, 20, CV_8UC4, Scalar(1, 2, 3, 4));
        Mat M = makeM(0, 0, 0, 1);
        Mat dst;
        ++gChecks;
        if (warpAffineNEON(src, dst, M, Size(10, 10), INTER_LINEAR,
                           BORDER_CONSTANT, Scalar())) {
            printf("FAIL 8UC4 должен вернуть false\n");
            ++gFailures;
        }
    }
    {
        Mat src(20, 20, CV_8UC3, Scalar(1, 2, 3));
        Mat M = makeM(0, 0, 0, 1);
        Mat dst;
        ++gChecks;
        if (warpAffineNEON(src, dst, M, Size(10, 10), INTER_LINEAR,
                           BORDER_REPLICATE, Scalar())) {
            printf("FAIL BORDER_REPLICATE должен вернуть false\n");
            ++gFailures;
        }
        ++gChecks;
        if (warpAffineNEON(src, dst, M, Size(10, 10), INTER_NEAREST,
                           BORDER_CONSTANT, Scalar())) {
            printf("FAIL INTER_NEAREST должен вернуть false\n");
            ++gFailures;
        }
        Mat bad = Mat::zeros(2, 3, CV_64F);
        ++gChecks;
        if (warpAffineNEON(src, dst, bad, Size(10, 10), INTER_LINEAR,
                           BORDER_CONSTANT, Scalar())) {
            printf("FAIL вырожденная матрица должна вернуть false\n");
            ++gFailures;
        }
    }

    // Заранее выделенный неcontinuous dst.
    {
        Mat src(40, 50, CV_8UC3);
        rng.fill(src, RNG::UNIFORM, 0, 256);
        Mat big(60, 80, CV_8UC3, Scalar(9, 9, 9));
        Mat dst = big(Rect(3, 4, 35, 27));
        Mat ref;
        Mat M = makeM(0.5, 1.0, -2.0, 1.0);
        warpAffine(src, ref, M, Size(35, 27), INTER_LINEAR, BORDER_CONSTANT, Scalar());
        const bool ok = warpAffineNEON(src, dst, M, Size(35, 27), INTER_LINEAR,
                                       BORDER_CONSTANT, Scalar(), 1);
        ++gChecks;
        if (!ok) { printf("FAIL non-continuous dst: false\n"); ++gFailures; }
        else check("non-continuous dst", ref, dst);
    }

    printf("\n%d checks, %d failures\n", gChecks, gFailures);
    return gFailures == 0 ? 0 : 1;
}
