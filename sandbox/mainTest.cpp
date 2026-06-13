// #include <opencv2/opencv.hpp>
// #include <opencv2/core/ocl.hpp>
// #include <iostream>
// #include <chrono>

// void testPerformance(int rows, int cols, int kernelSize, int iterations) {
//     std::cout << "\n========================================" << std::endl;
//     std::cout << "Тест: " << rows << "x" << cols << ", ядро: " << kernelSize << "x" << kernelSize 
//               << ", итераций: " << iterations << std::endl;
    
//     // Тест с UMat (OpenCL)
//     {
//         cv::ocl::setUseOpenCL(true);
        
//         cv::UMat src(rows, cols, CV_32FC1);
//         cv::UMat dst;
//         cv::randu(src, 0, 255);
        
//         // Прогрев
//         cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        
//         auto start = std::chrono::high_resolution_clock::now();
        
//         for (int i = 0; i < iterations; i++) {
//             cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
//         }
        
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
//         std::cout << "OpenCL (GPU): " << duration.count() << " мс" << std::endl;
//     }
    
//     // Тест с Mat (CPU)
//     {
//         cv::ocl::setUseOpenCL(false);
        
//         cv::Mat src(rows, cols, CV_32FC1);
//         cv::Mat dst;
//         cv::randu(src, 0, 255);
        
//         // Прогрев
//         cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        
//         auto start = std::chrono::high_resolution_clock::now();
        
//         for (int i = 0; i < iterations; i++) {
//             cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
//         }
        
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
//         std::cout << "CPU:          " << duration.count() << " мс" << std::endl;
//     }
// }

// cv::Mat getTransform(double dx, double dy, double da) {
//         cv::Mat T = cv::Mat::zeros(2, 3, CV_64F); // ИНИЦИАЛИЗИРУЕМ МАТРИЦУ
//         T.at<double>(0, 0) = cos(da);
//         T.at<double>(0, 1) = -sin(da);
//         T.at<double>(0, 2) = dx;
//         T.at<double>(1, 0) = sin(da);
//         T.at<double>(1, 1) = cos(da);
//         T.at<double>(1, 2) = dy;
//         return T;
// };


// void testPerformanceWrapAffine(int rows, int cols, cv::Mat T, int iterations) {
//     std::cout << "\n========================================" << std::endl;
//     std::cout << "Тест: " << rows << "x" << cols << ", итераций: " << iterations << std::endl;
    
//     // Тест с UMat (OpenCL)
//     {
//         cv::ocl::setUseOpenCL(true);
        
//         cv::UMat src(rows, cols, CV_32FC1);
//         cv::UMat dst;
//         cv::randu(src, 0, 255);
        
//         // Прогрев
//         //cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
//         cv::warpAffine(src, dst, T, cv::Size(src.rows, src.cols));
//         auto start = std::chrono::high_resolution_clock::now();
        
//         for (int i = 0; i < iterations; i++) {
//             cv::warpAffine(src, dst, T, cv::Size(src.rows, src.cols));
//         }
        
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
//         std::cout << "OpenCL (GPU): " << duration.count() << " мс" << std::endl;
//     }
    
//     // Тест с Mat (CPU)
//     {
//         cv::ocl::setUseOpenCL(false);
        
//         cv::Mat src(rows, cols, CV_32FC1);
//         cv::Mat dst;
//         cv::randu(src, 0, 255);
        
//         // Прогрев
//         //cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
//         cv::warpAffine(src, dst, T, cv::Size(src.rows, src.cols));
//         auto start = std::chrono::high_resolution_clock::now();
        
//         for (int i = 0; i < iterations; i++) {
//             cv::warpAffine(src, dst, T, cv::Size(src.rows, src.cols));
//         }
        
//         auto end = std::chrono::high_resolution_clock::now();
//         auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
//         std::cout << "CPU:          " << duration.count() << " мс" << std::endl;
//     }
// }


// int main() {
//     if (!cv::ocl::haveOpenCL()) {
//         std::cout << "OpenCL не доступен" << std::endl;
//         return -1;
//     }
    
//     cv::ocl::Device device = cv::ocl::Device::getDefault();
//     std::cout << "Устройство: " << device.name() << std::endl;
//     std::cout << "Тип: " << device.type() << std::endl;

//     // Тест с разными размерами ядра
//     std::cout << "\n\nТест с разными размерами ядра (изображение 2048x2048, 10 итераций):" << std::endl;
//     //testPerformance(2048, 2048, 5, 10);
//     //testPerformance(2048, 2048, 15, 10);
//     testPerformance(2048, 2048, 63, 10);
//     testPerformance(2048, 2048, 91, 10);
//     testPerformance(2048, 2048, 127, 10);

//     cv::Mat T = getTransform(10.0, 10.0, 2.0);
//     // Разные размеры данных для тестирования
//     testPerformanceWrapAffine(512, 512, T, 100);
    
//     // Тест с разными размерами ядра
//     std::cout << "\n\nТест с разными размерами изображения:" << std::endl;
//     //testPerformance(2048, 2048, 5, 10);
//     //testPerformance(2048, 2048, 15, 10);
//     testPerformanceWrapAffine(2048, 2048, T, 100);
//     testPerformanceWrapAffine(2048, 2048, T, 100);
//     testPerformanceWrapAffine(2048, 2048, T, 100);

//     return 0;
// }

#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <vector>
#include <future>  // Добавлено для std::promise и std::future

void testPerformance(int rows, int cols, int kernelSize, int iterations) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Тест: " << rows << "x" << cols << ", ядро: " << kernelSize << "x" << kernelSize 
              << ", итераций: " << iterations << std::endl;
    
    // Тест с UMat (OpenCL)
    {
        cv::ocl::setUseOpenCL(true);
        
        cv::UMat src(rows, cols, CV_32FC1);
        cv::UMat dst;
        cv::randu(src, 0, 255);
        
        // Прогрев
        cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "OpenCL (GPU): " << duration.count() << " мс" << std::endl;
    }
    
    // Тест с Mat (CPU)
    {
        cv::ocl::setUseOpenCL(false);
        
        cv::Mat src(rows, cols, CV_32FC1);
        cv::Mat dst;
        cv::randu(src, 0, 255);
        
        // Прогрев
        cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::GaussianBlur(src, dst, cv::Size(kernelSize, kernelSize), 0);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "CPU:          " << duration.count() << " мс" << std::endl;
    }
}

cv::Mat getTransform(double dx, double dy, double da) {
    cv::Mat T = cv::Mat::zeros(2, 3, CV_64F);
    T.at<double>(0, 0) = cos(da);
    T.at<double>(0, 1) = -sin(da);
    T.at<double>(0, 2) = dx;
    T.at<double>(1, 0) = sin(da);
    T.at<double>(1, 1) = cos(da);
    T.at<double>(1, 2) = dy;
    return T;
}

void testPerformanceWrapAffine(int rows, int cols, cv::Mat T, int iterations) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Тест: " << rows << "x" << cols << ", итераций: " << iterations << std::endl;
    
    // Тест с UMat (OpenCL)
    {
        cv::ocl::setUseOpenCL(true);
        
        cv::UMat src(rows, cols, CV_32FC1);
        cv::UMat dst;
        cv::randu(src, 0, 255);
        
        // Прогрев
        cv::warpAffine(src, dst, T, cv::Size(src.cols, src.rows)); // Исправлено: cols, rows
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::warpAffine(src, dst, T, cv::Size(src.cols, src.rows)); // Исправлено
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "OpenCL (GPU): " << duration.count() << " мс" << std::endl;
    }
    
    // Тест с Mat (CPU)
    {
        cv::ocl::setUseOpenCL(false);
        
        cv::Mat src(rows, cols, CV_32FC1);
        cv::Mat dst;
        cv::randu(src, 0, 255);
        
        // Прогрев
        cv::warpAffine(src, dst, T, cv::Size(src.cols, src.rows)); // Исправлено
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::warpAffine(src, dst, T, cv::Size(src.cols, src.rows)); // Исправлено
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "CPU:          " << duration.count() << " мс" << std::endl;
    }
}

// Класс для конвейерной обработки warpAffine
class WarpAffinePipeline {
private:
    struct Task {
        cv::UMat src;
        cv::UMat dst;
        cv::Mat transform;
        cv::Size dsize;
        int flags;
        int borderMode;
        cv::Scalar borderValue;
        std::promise<void> promise;  // Теперь включен <future>
    };

    std::queue<std::shared_ptr<Task>> tasks;
    std::mutex queueMutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::vector<std::thread> workerThreads;
    
    // Поток обработки
    void worker() {
        while (!stop) {
            std::shared_ptr<Task> task = nullptr;
            
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                condition.wait(lock, [this] { return !tasks.empty() || stop; });
                
                if (stop && tasks.empty()) {
                    return;
                }
                
                if (!tasks.empty()) {
                    task = tasks.front();
                    tasks.pop();
                }
            }
            
            if (task) {
                // Выполняем warpAffine
                cv::warpAffine(
                    task->src, 
                    task->dst, 
                    task->transform, 
                    task->dsize,
                    task->flags,
                    task->borderMode,
                    task->borderValue
                );
                
                // Синхронизируем с OpenCL очередью
                cv::ocl::finish();
                
                task->promise.set_value();
            }
        }
    }
    
    // Многопоточная обработка с конвейером
    void pipelineWorker(int threadId, int numThreads, 
                       std::vector<cv::UMat>& srcBatch,
                       std::vector<cv::UMat>& dstBatch,
                       const cv::Mat& T,
                       cv::Size dsize,
                       std::atomic<int>& completedCount,
                       int totalTasks) {
        
        for (int i = threadId; i < totalTasks; i += numThreads) {
            cv::warpAffine(srcBatch[i], dstBatch[i], T, dsize);
            
            // Частичная синхронизация для конвейера
            if (i % 4 == 0) { // Синхронизируем каждые 4 операции
                cv::ocl::finish();
            }
            
            completedCount++;
        }
    }

public:
    WarpAffinePipeline(int numThreads = 4) : stop(false) {
        // Создаем пул потоков
        for (int i = 0; i < numThreads; ++i) {
            workerThreads.emplace_back(&WarpAffinePipeline::worker, this);
        }
    }
    
    ~WarpAffinePipeline() {
        stop = true;
        condition.notify_all();
        
        for (auto& thread : workerThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    // Асинхронный запуск warpAffine
    std::future<void> asyncWarpAffine(
        cv::UMat src,
        cv::UMat dst,
        cv::Mat transform,
        cv::Size dsize,
        int flags = cv::INTER_LINEAR,
        int borderMode = cv::BORDER_CONSTANT,
        cv::Scalar borderValue = cv::Scalar()) {
        
        auto task = std::make_shared<Task>();
        task->src = src;
        task->dst = dst;
        task->transform = transform;
        task->dsize = dsize;
        task->flags = flags;
        task->borderMode = borderMode;
        task->borderValue = borderValue;
        
        auto future = task->promise.get_future();
        
        {
            std::lock_guard<std::mutex> lock(queueMutex);
            tasks.push(task);
        }
        
        condition.notify_one();
        return future;
    }
    
    // Пакетная обработка с конвейером
    void batchWarpAffine(
        std::vector<cv::UMat>& srcBatch,
        std::vector<cv::UMat>& dstBatch,
        const cv::Mat& T,
        cv::Size dsize) {
        
        if (srcBatch.size() != dstBatch.size()) {
            throw std::runtime_error("Batch sizes must match");
        }
        
        int numThreads = workerThreads.size();
        std::atomic<int> completedCount(0);
        int batchSize = srcBatch.size();
        
        // Создаем временный буфер для конвейеризации
        std::vector<std::thread> batchThreads;
        
        // Запускаем обработку в несколько потоков
        for (int t = 0; t < numThreads; ++t) {
            batchThreads.emplace_back(
                &WarpAffinePipeline::pipelineWorker, 
                this,
                t, numThreads,
                std::ref(srcBatch),
                std::ref(dstBatch),
                std::ref(T),
                dsize,
                std::ref(completedCount),
                batchSize
            );
        }
        
        // Ждем завершения всех потоков
        for (auto& thread : batchThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        
        // Финальная синхронизация
        cv::ocl::finish();
    }
    
    // Конвейер с двойной буферизацией
    void pipelinedWarpAffine(
        cv::UMat& src,
        cv::UMat& dst,
        const cv::Mat& T,
        cv::Size dsize,
        int iterations) {
        
        // Используем двойную буферизацию
        std::vector<cv::UMat> pingPongBuffer(2);
        src.copyTo(pingPongBuffer[0]);
        
        // Конвейер: загрузка -> обработка -> выгрузка
        for (int i = 0; i < iterations; ++i) {
            int currentBuffer = i % 2;
            int nextBuffer = (i + 1) % 2;
            
            // Асинхронно обрабатываем текущий буфер
            auto future = asyncWarpAffine(
                pingPongBuffer[currentBuffer],
                pingPongBuffer[nextBuffer],
                T,
                dsize
            );
            
            // Если не последняя итерация, готовим следующий кусок данных
            if (i < iterations - 1) {
                // Здесь можно подготовить следующую порцию данных
                // Например, скопировать следующий кадр
            }
            
            // Ждем завершения текущей операции
            future.wait();
        }
        
        // Копируем результат
        pingPongBuffer[iterations % 2].copyTo(dst);
    }
};

// Оптимизированная версия теста производительности с конвейеризацией
void testPerformanceWrapAffineOptimized(int rows, int cols, cv::Mat T, int iterations, int batchSize = 10) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Тест с конвейеризацией: " << rows << "x" << cols 
              << ", итераций: " << iterations << ", batch: " << batchSize << std::endl;
    
    cv::ocl::setUseOpenCL(true);
    
    // Создаем пул изображений для пакетной обработки
    std::vector<cv::UMat> srcBatch(batchSize);
    std::vector<cv::UMat> dstBatch(batchSize);
    
    for (int i = 0; i < batchSize; ++i) {
        srcBatch[i] = cv::UMat(rows, cols, CV_32FC1);
        cv::randu(srcBatch[i], 0, 255);
        dstBatch[i] = cv::UMat(rows, cols, CV_32FC1);
    }
    
    // Прогрев
    for (int i = 0; i < 3; ++i) {
        cv::warpAffine(srcBatch[0], dstBatch[0], T, cv::Size(cols, rows));
    }
    
    // Создаем конвейер
    WarpAffinePipeline pipeline(4); // 4 потока
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // Пакетная обработка
    int numBatches = iterations / batchSize;
    for (int b = 0; b < numBatches; ++b) {
        pipeline.batchWarpAffine(srcBatch, dstBatch, T, cv::Size(cols, rows));
    }
    
    // Обрабатываем остаток
    int remaining = iterations % batchSize;
    if (remaining > 0) {
        std::vector<cv::UMat> srcRemaining(srcBatch.begin(), srcBatch.begin() + remaining);
        std::vector<cv::UMat> dstRemaining(dstBatch.begin(), dstBatch.begin() + remaining);
        pipeline.batchWarpAffine(srcRemaining, dstRemaining, T, cv::Size(cols, rows));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "OpenCL GPU с конвейеризацией: " << duration.count() << " мс" << std::endl;
    
    // Для сравнения - обычный последовательный метод
    {
        cv::ocl::setUseOpenCL(true);
        
        cv::UMat src(rows, cols, CV_32FC1);
        cv::UMat dst;
        cv::randu(src, 0, 255);
        
        cv::warpAffine(src, dst, T, cv::Size(cols, rows));
        
        auto start_seq = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; i++) {
            cv::warpAffine(src, dst, T, cv::Size(cols, rows));
        }
        
        auto end_seq = std::chrono::high_resolution_clock::now();
        auto duration_seq = std::chrono::duration_cast<std::chrono::milliseconds>(end_seq - start_seq);
        
        std::cout << "OpenCL GPU последовательно: " << duration_seq.count() << " мс" << std::endl;
        std::cout << "Ускорение: " << (double)duration_seq.count() / duration.count() << "x" << std::endl;
    }
}

// Тест с двойной буферизацией
void testDoubleBuffering(int rows, int cols, cv::Mat T, int iterations) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "Тест с двойной буферизацией: " << rows << "x" << cols 
              << ", итераций: " << iterations << std::endl;
    
    cv::ocl::setUseOpenCL(true);
    
    cv::UMat src(rows, cols, CV_32FC1);
    cv::UMat dst;
    cv::randu(src, 0, 255);
    
    // Прогрев
    cv::warpAffine(src, dst, T, cv::Size(cols, rows));
    
    WarpAffinePipeline pipeline(2);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    pipeline.pipelinedWarpAffine(src, dst, T, cv::Size(cols, rows), iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "С двойной буферизацией: " << duration.count() << " мс" << std::endl;
}

int main() {
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL не доступен" << std::endl;
        return -1;
    }
    
    // Включаем OpenCL и настраиваем
    cv::ocl::setUseOpenCL(true);
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    std::cout << "Устройство: " << device.name() << std::endl;
    std::cout << "Тип: " << device.type() << std::endl;
    
    cv::Mat T = getTransform(10.0, 10.0, 2.0);
    
    // Тесты с разными размерами
    std::cout << "\n\nТестирование оптимизированного warpAffine:" << std::endl;
    
    // Малые изображения
    testPerformanceWrapAffineOptimized(512, 512, T, 200, 10);
    
    // Средние изображения
    testPerformanceWrapAffineOptimized(1024, 1024, T, 100, 10);
    
    // Большие изображения
    testPerformanceWrapAffineOptimized(2048, 2048, T, 50, 10);
    
    // Тест с двойной буферизацией
    testDoubleBuffering(2048, 2048, T, 10);
    
    return 0;
}