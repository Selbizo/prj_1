#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include "src/warpAffine_neon_optimized.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

/**
 * Бенчмарк для тестирования производительности NEON warpAffine оптимизации
 * 
 * Использование:
 *   g++ -O3 -mfpu=neon -march=armv7-a -fopenmp `pkg-config --cflags --libs opencv4` \
 *       benchmark_warpaffine.cpp -o benchmark_warpaffine
 *   ./benchmark_warpaffine
 */

struct BenchmarkResult {
    Size frameSize;
    int numIterations;
    double timeStdMs;      // Стандартная warpAffine
    double timeNeonMs;     // NEON оптимизированная
    double speedup;        // Ускорение в раз
    
    void print() const {
        cout << "\n========== РЕЗУЛЬТАТЫ БЕНЧМАРКА ==========" << endl;
        cout << "Разрешение: " << frameSize.width << "x" << frameSize.height << endl;
        cout << "Итераций: " << numIterations << endl;
        cout << "\nВремя выполнения:" << endl;
        cout << "  Стандартная:  " << fixed << setprecision(2) << timeStdMs << " мс" << endl;
        cout << "  NEON версия:  " << timeNeonMs << " мс" << endl;
        cout << "\nУскорение: " << fixed << setprecision(2) << speedup << "x" << endl;
        cout << "Улучшение: " << fixed << setprecision(1) << (speedup - 1.0) * 100 << "%" << endl;
        cout << "=========================================" << endl;
    }
};

Mat createTestFrame(Size size, int channels = 3) {
    Mat frame = Mat::zeros(size, CV_8UC3);
    
    // Добавить некоторый контент для реалистичного теста
    randu(frame, Scalar(0, 0, 0), Scalar(256, 256, 256));
    
    // Нарисовать несколько линий и прямоугольников
    for (int i = 0; i < 10; i++) {
        Point pt1(rand() % size.width, rand() % size.height);
        Point pt2(rand() % size.width, rand() % size.height);
        line(frame, pt1, pt2, Scalar(255, 255, 255), 2);
    }
    
    return frame;
}

Mat createRandomTransformMatrix(double maxTranslation = 10.0, double maxRotation = 0.1) {
    Mat M = Mat::eye(2, 3, CV_64F);
    
    // Случайный сдвиг
    M.at<double>(0, 2) = (rand() % (int)(2 * maxTranslation)) - maxTranslation;
    M.at<double>(1, 2) = (rand() % (int)(2 * maxTranslation)) - maxTranslation;
    
    // Случайный поворот (очень маленький для стабилизации)
    double angle = (rand() % (int)(2 * maxRotation * 1000)) / 1000.0 - maxRotation;
    double cosA = cos(angle);
    double sinA = sin(angle);
    M.at<double>(0, 0) = cosA;
    M.at<double>(0, 1) = -sinA;
    M.at<double>(1, 0) = sinA;
    M.at<double>(1, 1) = cosA;
    
    return M;
}

BenchmarkResult benchmarkWarpAffine(Size frameSize, int numIterations = 10) {
    cout << "\nТестирование разрешения " << frameSize.width << "x" << frameSize.height << "..." << endl;
    
    // Создать тестовый кадр
    Mat frame = createTestFrame(frameSize);
    Mat result_std, result_neon;
    
    BenchmarkResult result;
    result.frameSize = frameSize;
    result.numIterations = numIterations;
    
    // === СТАНДАРТНАЯ ВЕРСИЯ ===
    cout << "  Запуск стандартной warpAffine... ";
    cout.flush();
    
    auto startStd = high_resolution_clock::now();
    
    for (int i = 0; i < numIterations; i++) {
        Mat M = createRandomTransformMatrix();
        cv::warpAffine(frame, result_std, M, frameSize, INTER_LINEAR, BORDER_CONSTANT);
    }
    
    auto endStd = high_resolution_clock::now();
    result.timeStdMs = duration<double, milli>(endStd - startStd).count();
    cout << "OK (" << fixed << setprecision(1) << result.timeStdMs << " мс)" << endl;
    
    // === NEON ВЕРСИЯ ===
    cout << "  Запуск NEON warpAffine... ";
    cout.flush();
    
    auto startNeon = high_resolution_clock::now();
    
    for (int i = 0; i < numIterations; i++) {
        Mat M = createRandomTransformMatrix();
        WarpAffineNeonOptimized::warpAffine(frame, result_neon, M, frameSize, 
                                           INTER_LINEAR, BORDER_CONSTANT, Scalar(), true);
    }
    
    auto endNeon = high_resolution_clock::now();
    result.timeNeonMs = duration<double, milli>(endNeon - startNeon).count();
    cout << "OK (" << fixed << setprecision(1) << result.timeNeonMs << " мс)" << endl;
    
    // === ВЫЧИСЛЕНИЕ УСКОРЕНИЯ ===
    result.speedup = result.timeStdMs / result.timeNeonMs;
    
    return result;
}

int main() {
    cout << "=========================================" << endl;
    cout << "БЕНЧМАРК NEON warpAffine оптимизации" << endl;
    cout << "=========================================" << endl;
    
    // Информация о системе
    #ifdef _OPENMP
    cout << "OpenMP: Да, " << omp_get_num_procs() << " потоков доступно" << endl;
    #else
    cout << "OpenMP: Нет" << endl;
    #endif
    
    cout << "OpenCV версия: " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << endl;
    cout << "Компилятор оптимизации: -O3 -mfpu=neon -march=armv7-a" << endl;
    
    // Прогрев (warm-up)
    cout << "\nПрогрев процессора... ";
    cout.flush();
    Mat warmupFrame = createTestFrame(Size(640, 480));
    for (int i = 0; i < 10; i++) {
        Mat M = createRandomTransformMatrix();
        Mat dummy;
        cv::warpAffine(warmupFrame, dummy, M, warmupFrame.size());
        WarpAffineNeonOptimized::warpAffine(warmupFrame, dummy, M, warmupFrame.size());
    }
    cout << "OK" << endl;
    
    // === ОСНОВНЫЕ ТЕСТЫ ===
    vector<BenchmarkResult> results;
    
    // Маленькое разрешение (где NEON может не использоваться)
    results.push_back(benchmarkWarpAffine(Size(640, 480), 20));
    
    // Стандартное разрешение
    results.push_back(benchmarkWarpAffine(Size(1280, 720), 10));
    
    // HD разрешение (целевое)
    results.push_back(benchmarkWarpAffine(Size(1920, 1080), 5));
    
    // === ВЫВОД РЕЗУЛЬТАТОВ ===
    cout << "\n\n========== ИТОГОВЫЕ РЕЗУЛЬТАТЫ ==========" << endl;
    cout << "\n┌─────────────┬──────────────┬──────────────┬────────────┐" << endl;
    cout << "│ Разрешение  │ Стандартная  │ NEON версия  │ Ускорение  │" << endl;
    cout << "├─────────────┼──────────────┼──────────────┼────────────┤" << endl;
    
    for (const auto& r : results) {
        printf("│ %4dx%-4d   │ %8.2f мс   │ %8.2f мс   │ %6.2fx    │\n",
               r.frameSize.width, r.frameSize.height, 
               r.timeStdMs, r.timeNeonMs, r.speedup);
    }
    
    cout << "└─────────────┴──────────────┴──────────────┴────────────┘" << endl;
    
    // === АНАЛИЗ ===
    double avgSpeedup = 0;
    for (const auto& r : results) {
        avgSpeedup += r.speedup;
    }
    avgSpeedup /= results.size();
    
    cout << "\nСредний прирост производительности: " << fixed << setprecision(2) 
         << avgSpeedup << "x (" << (avgSpeedup - 1.0) * 100 << "%)" << endl;
    
    // === ОЦЕНКА ===
    cout << "\nОценка производительности:" << endl;
    if (avgSpeedup >= 1.5) {
        cout << "✓ Отличный результат! NEON оптимизация очень эффективна." << endl;
    } else if (avgSpeedup >= 1.2) {
        cout << "✓ Хороший результат. NEON дает заметное улучшение." << endl;
    } else if (avgSpeedup >= 1.0) {
        cout << "○ Нейтрально. NEON примерно равна стандартной версии." << endl;
    } else {
        cout << "✗ NEON медленнее стандартной версии (может быть из-за overhead)." << endl;
    }
    
    // === РЕКОМЕНДАЦИИ ===
    cout << "\nРекомендации:" << endl;
    cout << "1. Для производства используйте NEON с разрешением >= 1024x768" << endl;
    cout << "2. Проверьте, что все 4 ядра ARM задействованы (top)" << endl;
    cout << "3. Убедитесь в достаточном охлаждении процессора" << endl;
    cout << "4. Для максимума производительности отключите frequency scaling" << endl;
    
    cout << "\n=========================================" << endl;
    
    return 0;
}
