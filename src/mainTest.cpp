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
    
//     // Разные размеры данных для тестирования
//     testPerformance(512, 512, 15, 100);
//     //testPerformance(1024, 1024, 15, 50);
//     //testPerformance(2048, 2048, 15, 20);
//     //testPerformance(4096, 4096, 15, 10);
//     //testPerformance(8192, 8192, 15, 10);
    
//     // Тест с разными размерами ядра
//     std::cout << "\n\nТест с разными размерами ядра (изображение 2048x2048, 10 итераций):" << std::endl;
//     //testPerformance(2048, 2048, 5, 10);
//     //testPerformance(2048, 2048, 15, 10);
//     testPerformance(2048, 1024, 39, 5);
//     testPerformance(2048, 1024, 49, 5);
//     testPerformance(2048, 1024, 63, 5);

//     cv::Mat T = getTransform(10.0, 10.0, 2.0);
//     // Разные размеры данных для тестирования
//     testPerformanceWrapAffine(512, 512, T, 100);
    
//     // Тест с разными размерами ядра
//     std::cout << "\n\nТест с разными размерами изображения:" << std::endl;
//     //testPerformance(2048, 2048, 5, 10);
//     //testPerformance(2048, 2048, 15, 10);
//     testPerformanceWrapAffine(2048, 2048, T, 100);
//     testPerformanceWrapAffine(2048, 1024, T, 100);
//     testPerformanceWrapAffine(1024, 1024, T, 100);

//     return 0;
// }


#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <vector>
#include <utility> // для std::pair

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    
    // Правильный способ получения доступных бэкендов в OpenCV 4.12.0
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = cv::dnn::getAvailableBackends();
    std::cout << "\nДоступные бэкенды:" << std::endl;
    for (const auto& backend_pair : backends) {
        cv::dnn::Backend backend = backend_pair.first;
        cv::dnn::Target target = backend_pair.second;
        
        std::cout << "  Backend: " << backend;
        if (backend == cv::dnn::DNN_BACKEND_TIMVX) {
            std::cout << " (TIMVX) ⭐";
        } else if (backend == cv::dnn::DNN_BACKEND_OPENCV) {
            std::cout << " (OpenCV)";
        } else if (backend == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE) {
            std::cout << " (Inference Engine)";
        } else if (backend == cv::dnn::DNN_BACKEND_HALIDE) {
            std::cout << " (Halide)";
        } else if (backend == cv::dnn::DNN_BACKEND_VKCOM) {
            std::cout << " (VKCOM)";
        } else if (backend == cv::dnn::DNN_BACKEND_CUDA) {
            std::cout << " (CUDA)";
        } else {
            std::cout << " (Unknown)";
        }
        
        std::cout << " | Target: " << target;
        if (target == cv::dnn::DNN_TARGET_NPU) {
            std::cout << " (NPU) ⚡";
        } else if (target == cv::dnn::DNN_TARGET_CPU) {
            std::cout << " (CPU)";
        } else if (target == cv::dnn::DNN_TARGET_OPENCL) {
            std::cout << " (OpenCL)";
        } else if (target == cv::dnn::DNN_TARGET_OPENCL_FP16) {
            std::cout << " (OpenCL FP16)";
        } else if (target == cv::dnn::DNN_TARGET_MYRIAD) {
            std::cout << " (Myriad)";
        } else if (target == cv::dnn::DNN_TARGET_FPGA) {
            std::cout << " (FPGA)";
        } else if (target == cv::dnn::DNN_TARGET_CUDA) {
            std::cout << " (CUDA)";
        } else if (target == cv::dnn::DNN_TARGET_CUDA_FP16) {
            std::cout << " (CUDA FP16)";
        } else if (target == cv::dnn::DNN_TARGET_HDDL) {
            std::cout << " (HDDL)";
        }
        std::cout << std::endl;
    }
    
    // Проверка TIM-VX - теперь нужно указывать и бэкенд, и цель
    std::cout << "\n--- Проверка TIM-VX ---" << std::endl;
    
    bool timvx_found = false;
    for (const auto& backend_pair : backends) {
        if (backend_pair.first == cv::dnn::DNN_BACKEND_TIMVX) {
            timvx_found = true;
            if (backend_pair.second == cv::dnn::DNN_TARGET_NPU) {
                std::cout << "✅ TIM-VX с NPU доступен!" << std::endl;
            } else {
                std::cout << "✅ TIM-VX доступен (цель: " << backend_pair.second << ")" << std::endl;
            }
        }
    }
    
    if (!timvx_found) {
        std::cout << "❌ TIM-VX не найден в списке доступных бэкендов" << std::endl;
    }
    
    // Альтернативный способ проверки через getAvailableTargets
    try {
        std::cout << "\n--- Проверка через getAvailableTargets ---" << std::endl;
        std::vector<cv::dnn::Target> targets = cv::dnn::getAvailableTargets(cv::dnn::DNN_BACKEND_TIMVX);
        if (!targets.empty()) {
            std::cout << "✅ TIM-VX доступен! Найдено целей: " << targets.size() << std::endl;
            for (const auto& target : targets) {
                std::cout << "  Target: " << target;
                if (target == cv::dnn::DNN_TARGET_NPU) {
                    std::cout << " (NPU) ⚡";
                }
                std::cout << std::endl;
            }
        } else {
            std::cout << "❌ TIM-VX не доступен через getAvailableTargets" << std::endl;
        }
    } catch (const cv::Exception& e) {
        std::cout << "⚠️ Ошибка при вызове getAvailableTargets: " << e.what() << std::endl;
    }
    
    return 0;
}