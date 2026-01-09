#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <chrono>

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

int main() {
    if (!cv::ocl::haveOpenCL()) {
        std::cout << "OpenCL не доступен" << std::endl;
        return -1;
    }
    
    cv::ocl::Device device = cv::ocl::Device::getDefault();
    std::cout << "Устройство: " << device.name() << std::endl;
    std::cout << "Тип: " << device.type() << std::endl;
    
    // Разные размеры данных для тестирования
    testPerformance(512, 512, 15, 100);
    testPerformance(1024, 1024, 15, 50);
    testPerformance(2048, 2048, 15, 20);
    testPerformance(4096, 4096, 15, 10);
    testPerformance(8192, 8192, 15, 10);
    
    // Тест с разными размерами ядра
    std::cout << "\n\nТест с разными размерами ядра (изображение 2048x2048, 10 итераций):" << std::endl;
    testPerformance(2048, 2048, 5, 10);
    testPerformance(2048, 2048, 15, 10);
    testPerformance(2048, 2048, 31, 10);
    testPerformance(2048, 2048, 63, 10);
    
    return 0;
}