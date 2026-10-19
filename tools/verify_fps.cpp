// verify_fps.cpp — независимое доказательство частоты кадров через тот же
// интерфейс, что и приложение: cv::VideoCapture на /dev/video0, raw YUYV 1920x1080.
//
// Отвечает на два вопроса:
//   1) «VideoCapture должен стоить 8 мс — как может быть 120 fps?»
//      Фаза A (пустой цикл) показывает, что 8.3 мс — это ПЕРИОД СЕНСОРА:
//      read() блокируется на VSYNC. Внутри busy-цикла кадр уже готов в очереди
//      драйвера, и read() стоит меньше (фаза B).
//   2) «Может, fps завышен дубликатами буферов?»
//      Каждый кадр хешируется (FNV-1a), считается число уникальных. Если
//      уникальных == всего, дубликатов нет и fps честный.
//
// Сборка: см. tools/README.md
// Запуск: /tmp/verify_fps [секунды]        (по умолчанию 10 на фазу)

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
using namespace cv;
using namespace std;
using clk = chrono::steady_clock;

static double ms(clk::time_point a, clk::time_point b) {
    return chrono::duration_cast<chrono::nanoseconds>(b - a).count() / 1e6;
}

// FNV-1a по содержимому кадра с шагом 16 байт (как в lightdiag) — дёшево даже
// на 120 fps и достаточно, чтобы отличить разные кадры от повторов буфера.
static uint64_t fnv(const Mat& m) {
    const uint8_t* b = m.data;
    size_t n = m.total() * m.elemSize();
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i += 16) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

static double median(vector<double> v) {
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double pct(vector<double> v, double p) {
    if (v.empty()) return 0;
    sort(v.begin(), v.end());
    return v[min((size_t)(v.size() * p), v.size() - 1)];
}

static VideoCapture openCam() {
    VideoCapture c;
    c.open("/dev/video0", CAP_V4L2);
    if (!c.isOpened()) return c;
    c.set(CAP_PROP_FRAME_WIDTH, 1920);
    c.set(CAP_PROP_FRAME_HEIGHT, 1080);
    c.set(CAP_PROP_FOURCC, VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    c.set(CAP_PROP_CONVERT_RGB, 0);   // сырой YUYV, как в estimate-only
    return c;
}

// Плоский raw YUYV (Nx1 8UC1) -> 2-канальный 1080x1920.
static Mat unflatten(const Mat& raw, int rows, int cols) {
    if (CV_MAT_CN(raw.type()) == 2) return raw;
    if (raw.total() == (size_t)rows * cols * 2) return raw.reshape(2, rows);
    return Mat();
}

struct Stats {
    long long frames = 0, unique = 0, dup = 0;
    double wallFps = 0;
    vector<double> iv;    // интервалы между кадрами, мс
    vector<double> readv; // время read(), мс
    vector<double> prepv; // время prep(), мс
};

int main(int argc, char** argv) {
    setbuf(stdout, NULL);
    double secs = (argc > 1) ? atof(argv[1]) : 10.0;
    const int W = 1920, H = 1080;
    const int COMPRESSION = 5;                 // как в приложении
    Size comp(W / COMPRESSION, H / COMPRESSION);

    printf("=== ПРОВЕРКА FPS: cv::VideoCapture /dev/video0, raw YUYV 1920x1080 ===\n");
    printf("фаза по %.0f с, tracking-размер %dx%d (compressionConfig=%d)\n\n",
           secs, comp.width, comp.height, COMPRESSION);

    // ---------------- Фаза A: пустой цикл, только read() ----------------
    {
        VideoCapture c = openCam();
        if (!c.isOpened()) { printf("НЕ ОТКРЫЛСЯ /dev/video0\n"); return 1; }
        Mat raw;
        for (int i = 0; i < 30; i++) c.read(raw);   // прогрев

        Stats s;
        vector<uint64_t> hashes;
        hashes.reserve(2000);
        auto tStart = clk::now();
        auto prevFrame = tStart;
        bool first = true;
        while (ms(tStart, clk::now()) < secs * 1000.0) {
            auto a = clk::now();
            if (!c.read(raw)) break;
            auto b = clk::now();
            s.readv.push_back(ms(a, b));
            if (!first) s.iv.push_back(ms(prevFrame, b));
            prevFrame = b; first = false;
            hashes.push_back(fnv(raw));
            s.frames++;
        }
        double dt = ms(tStart, clk::now());
        s.wallFps = s.frames * 1000.0 / dt;
        {
            vector<uint64_t> u = hashes;
            sort(u.begin(), u.end());
            u.erase(unique(u.begin(), u.end()), u.end());
            s.unique = (long long)u.size();
            s.dup = s.frames - s.unique;
        }
        printf("[Фаза A] пустой цикл: только cap.read()\n");
        printf("  кадров: %lld   уникальных: %lld   дубликатов буфера: %lld\n",
               s.frames, s.unique, s.dup);
        printf("  wall-clock:            %.2f fps   (%.3f мс на кадр)\n",
               s.wallFps, dt / s.frames);
        printf("  интервалы между кадрами: min %.2f / median %.2f / p95 %.2f / max %.2f мс\n",
               *min_element(s.iv.begin(), s.iv.end()), median(s.iv),
               pct(s.iv, 0.95), *max_element(s.iv.begin(), s.iv.end()));
        printf("  время самого read():    median %.2f мс\n", median(s.readv));
        printf("  => при пустом цикле read блокируется на период сенсора (~%.1f мс).\n\n",
               median(s.iv));
        c.release();
    }

    // ---------------- Фаза B: путь приложения (read + prep) ----------------
    {
        VideoCapture c = openCam();
        Mat raw, smallYUYV, gray;
        for (int i = 0; i < 30; i++) c.read(raw);

        Stats s;
        vector<uint64_t> hashes;
        hashes.reserve(2000);
        auto tStart = clk::now();
        auto prevFrame = tStart;
        bool first = true;
        while (ms(tStart, clk::now()) < secs * 1000.0) {
            auto a = clk::now();
            if (!c.read(raw)) break;
            auto b = clk::now();
            // prep ровно как в video_stabilizer.cpp (rawCapture, вариант J):
            // resize 2-канального YUYV по INTER_LINEAR -> YUYV2GRAY.
            Mat yuyv = unflatten(raw, H, W);
            Size conv(comp.width & ~1, comp.height);
            if (!yuyv.empty()) {
                resize(yuyv, smallYUYV, conv, 0, 0, INTER_LINEAR);
                cvtColor(smallYUYV, gray, COLOR_YUV2GRAY_YUY2);
            }
            auto d = clk::now();
            s.readv.push_back(ms(a, b));
            s.prepv.push_back(ms(b, d));
            if (!first) s.iv.push_back(ms(prevFrame, d));
            prevFrame = d; first = false;
            hashes.push_back(fnv(raw));
            s.frames++;
        }
        double dt = ms(tStart, clk::now());
        s.wallFps = s.frames * 1000.0 / dt;
        {
            vector<uint64_t> u = hashes;
            sort(u.begin(), u.end());
            u.erase(unique(u.begin(), u.end()), u.end());
            s.unique = (long long)u.size();
            s.dup = s.frames - s.unique;
        }
        double loopMed = median(s.iv);
        printf("[Фаза B] путь приложения: read + resize YUYV->GRAY (вариант J)\n");
        printf("  кадров: %lld   уникальных: %lld   дубликатов буфера: %lld\n",
               s.frames, s.unique, s.dup);
        printf("  wall-clock:            %.2f fps   (%.3f мс на кадр)\n",
               s.wallFps, dt / s.frames);
        printf("  время read():          median %.2f мс   (в busy-цикле кадр уже в очереди)\n",
               median(s.readv));
        printf("  время prep():          median %.2f мс\n", median(s.prepv));
        printf("  полный цикл на кадр:   median %.2f мс\n", loopMed);
        printf("  интервалы между кадрами: min %.2f / median %.2f / p95 %.2f / max %.2f мс\n",
               *min_element(s.iv.begin(), s.iv.end()), loopMed,
               pct(s.iv, 0.95), *max_element(s.iv.begin(), s.iv.end()));
        printf("  бюджет 120 fps = 8.33 мс/кадр; запас = %.2f мс\n\n",
               8.33 - loopMed);
        c.release();
    }

    // ---------------- Фаза C: строгая проверка уникальности ----------------
    // memcmp клонированного предыдущего кадра: если драйвер/OpenCV вернули
    // тот же буфер дважды подряд, содержимое совпадёт побайтово. Хеш — по всем
    // байтам кадра (шаг 1), поэтому коллизии исключены.
    {
        VideoCapture c = openCam();
        Mat raw, prev;
        for (int i = 0; i < 30; i++) c.read(raw);
        long long frames = 0, consecIdent = 0;
        vector<uint64_t> hashes;
        hashes.reserve(2000);
        auto tStart = clk::now();
        while (ms(tStart, clk::now()) < 5.0 * 1000.0) {
            if (!c.read(raw)) break;
            size_t n = raw.total() * raw.elemSize();
            if (!prev.empty() && prev.total() * prev.elemSize() == n &&
                memcmp(raw.data, prev.data, n) == 0) consecIdent++;
            const uint8_t* b = raw.data;
            uint64_t h = 1469598103934665603ULL;
            for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
            hashes.push_back(h);
            raw.copyTo(prev);
            frames++;
        }
        sort(hashes.begin(), hashes.end());
        hashes.erase(unique(hashes.begin(), hashes.end()), hashes.end());
        long long uniqueFull = (long long)hashes.size();
        printf("[Фаза C] строгая уникальность (полный хеш шаг=1 + memcmp соседних)\n");
        printf("  кадров: %lld   уникальных по полному хешу: %lld   дубликатов: %lld\n",
               frames, uniqueFull, frames - uniqueFull);
        printf("  кадров, побайтово равных предыдущему: %lld  (0 = драйвер НЕ повторяет буфер)\n\n",
               consecIdent);
        c.release();
    }

    printf("ВЫВОД: если дубликатов буфера нет, а медиана интервала ~8.3 мс —\n");
    printf("сенсор реально идёт на ~120 Гц, и fps не завышен счётом повторов.\n");
    return 0;
}
