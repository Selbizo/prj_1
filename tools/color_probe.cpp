// Диагностика цветового пути MIPI-камеры OS08A10.
//
// Что делает:
//   1) открывает /dev/video0 через CAP_V4L2 и просит 1920x1080 YUYV;
//   2) печатает фактический FOURCC и размер;
//   3) читает кадр штатным путём OpenCV (CONVERT_RGB=1) и сохраняет PNG;
//   4) читает сырой буфер (CONVERT_RGB=0), сохраняет варианты YUY2/UYVY;
//   5) печатает средние BGR и разницу между штатным и ручным путём.
//
// Сборка (из корня проекта):
//   OC=/usr/local/opencv-4.12.0
//   g++ -O2 -std=gnu++17 -isystem $OC/include/opencv4 tools/color_probe.cpp \
//       -o /tmp/color_probe -L$OC/lib -Wl,-rpath,$OC/lib \
//       -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_imgcodecs
//
// Запуск:
//   /tmp/color_probe [--device /dev/video0] [--out /tmp/opencode/color]

#include <opencv2/opencv.hpp>
#include <cstdio>
#include <string>
#include <vector>

using namespace cv;
using namespace std;

static string fourccStr(int v) {
    char s[5] = { char(v & 255), char((v >> 8) & 255),
                  char((v >> 16) & 255), char((v >> 24) & 255), 0 };
    return string(s);
}

static void printMean(const Mat& bgr, const char* tag) {
    Scalar m = mean(bgr);
    printf("%-22s %dx%d ch=%d BGR mean=%.2f %.2f %.2f\n",
           tag, bgr.cols, bgr.rows, bgr.channels(), m[0], m[1], m[2]);
}

int main(int argc, char** argv) {
    string device = "/dev/video0";
    string outDir = "/tmp/opencode/color";
    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        if (a == "--device" && i + 1 < argc) device = argv[++i];
        else if (a == "--out" && i + 1 < argc) outDir = argv[++i];
        else if (a == "-h" || a == "--help") {
            printf("Использование: %s [--device /dev/videoN] [--out DIR]\n", argv[0]);
            return 0;
        }
    }

    VideoCapture cap(device, CAP_V4L2);
    if (!cap.isOpened()) {
        fprintf(stderr, "Не удалось открыть %s\n", device.c_str());
        return 1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 1920);
    cap.set(CAP_PROP_FRAME_HEIGHT, 1080);
    cap.set(CAP_PROP_FOURCC, VideoWriter::fourcc('Y', 'U', 'Y', 'V'));

    printf("Запрошено YUYV 1920x1080; фактически FOURCC=%s %dx%d\n",
           fourccStr(static_cast<int>(cap.get(CAP_PROP_FOURCC))).c_str(),
           static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH)),
           static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)));

    Mat opencvBgr;
    for (int i = 0; i < 5; ++i) cap.read(opencvBgr);
    if (opencvBgr.empty()) {
        fprintf(stderr, "Не удалось прочитать кадр (CONVERT_RGB=1)\n");
        return 1;
    }
    string p1 = outDir + "/opencv_bgr.png";
    imwrite(p1, opencvBgr);
    printMean(opencvBgr, "CONVERT_RGB=1 (OpenCV)");

    cap.set(CAP_PROP_CONVERT_RGB, 0);
    Mat raw;
    for (int i = 0; i < 5; ++i) cap.read(raw);
    if (raw.empty()) {
        fprintf(stderr, "Не удалось прочитать сырой кадр (CONVERT_RGB=0)\n");
        return 1;
    }
    printf("Сырой буфер: %dx%d ch=%d bytes=%zu (ожидается %d байт)\n",
           raw.cols, raw.rows, raw.channels(),
           raw.total() * raw.elemSize(), 1920 * 1080 * 2);

    Mat flat = raw;
    if (flat.channels() == 1 && flat.total() % (1920u * 2u) == 0) {
        int rows = static_cast<int>(flat.total() / (1920u * 2u));
        if (rows > 0) flat = flat.reshape(2, rows);
    }
    if (flat.channels() != 2) {
        fprintf(stderr, "Ожидался 2-канальный YUYV, получено ch=%d\n", flat.channels());
        return 1;
    }

    Mat rawYuy2, rawUyvy;
    cvtColor(flat, rawYuy2, COLOR_YUV2BGR_YUY2);
    cvtColor(flat, rawUyvy, COLOR_YUV2BGR_UYVY);
    imwrite(outDir + "/raw_yuy2.png", rawYuy2);
    imwrite(outDir + "/raw_uyvy.png", rawUyvy);
    printMean(rawYuy2, "raw->BGR (YUY2)");
    printMean(rawUyvy, "raw->BGR (UYVY)");

    // Средние Y/U/V — полезны для проверки диапазона (limited/full).
    double sy = 0, su = 0, sv = 0;
    size_t n = 0;
    for (int y = 0; y < flat.rows; ++y) {
        const uchar* p = flat.ptr<uchar>(y);
        for (int x = 0; x < flat.cols; x += 2) {
            sy += p[2 * x] + p[2 * x + 2];
            su += p[2 * x + 1];
            sv += p[2 * x + 3];
            n += 2;
        }
    }
    printf("Y/U/V mean = %.2f / %.2f / %.2f\n", sy / n, su / (n / 2), sv / (n / 2));

    Mat diff;
    absdiff(opencvBgr, rawYuy2, diff);
    Scalar d = mean(diff);
    printf("mean|OpenCV - raw YUY2| BGR = %.2f %.2f %.2f\n", d[0], d[1], d[2]);
    printf("\nСохранено:\n  %s\n  %s/raw_yuy2.png\n  %s/raw_uyvy.png\n", p1.c_str(), outDir.c_str(), outDir.c_str());
    printf("Примечание: переключение CONVERT_RGB может менять экспозицию/формат потока,\n"
           "поэтому для сравнения цветовых формул фиксируйте AE/AWB через v4l2-ctl.\n");
    return 0;
}
