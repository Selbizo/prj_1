#include "config/config.h"
#include "config/structures.h"
#include "stabilizer/video_stabilizer.h"
#include "gpu/warp_affine.h"
#include "utils/helpers.h"

#include <iostream>
#include <thread>
#include <chrono>
#include <string>

using namespace std;

// ========================= ГЛАВНАЯ ФУНКЦИЯ =========================

int main() {
    cout << "========================================" << endl;
    cout << " MULTI-THREADED VIDEO STABILIZER OPENCL " << endl;
    cout << "========================================" << endl;
    
    cv::ocl::setUseOpenCL(USE_OPENCL);
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    
    // Обнаружить возможности GPU
    cout << "\n[ИНИЦИАЛИЗАЦИЯ GPU]" << endl;
    GPUCapabilities gpuCaps = detectGPUCapabilities();
    bool useFP16 = gpuCaps.supportsFP16 || gpuCaps.supportsHalfType;
    if (!useFP16) {
        cout << "[GPU] Предупреждение: FP16 недоступен, используется стандартная обработка" << endl;
    }
    cout << endl;
    
    // Выбор режима работы
    cout << "Выберите режим работы:" << endl;
    cout << "0. Использовать IP-камеру" << endl;
    cout << "1. Использовать камеру" << endl;
    cout << "2. Читать кадры из папки PXL_3" << endl;
    cout << "3. Читать кадры из папки PXL_4K" << endl;
    cout << "4. Читать видео из файла" << endl;
    cout << "Введите 0, 1, 2, 3 или 4: ";
    
    int choice;
    cin >> choice;
    
    bool useVideo = true;
    string imageFolderPath;
    
    if (choice == 2) {
        useVideo = false;
        cout << "Введите путь к папке с кадрами: ";
        cin.ignore();
        getline(cin, imageFolderPath);
        imageFolderPath = normalizePath(imageFolderPath);
        if (imageFolderPath.empty()) {
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_3/";
        }
        cout << "Путь к кадрам: " << imageFolderPath << endl;
    } else if (choice == 3) {
        useVideo = false;
        cout << "Введите путь к папке с кадрами: ";
        cin.ignore();
        getline(cin, imageFolderPath);
        imageFolderPath = normalizePath(imageFolderPath);
        if (imageFolderPath.empty()) {
            imageFolderPath = "/home/pi/opencv_projects/videos/PXL_4K/";
        }
        cout << "Путь к кадрам: " << imageFolderPath << endl;
    } else if (choice == 0) {
        cout << "Введите URL IP-камеры (Enter = по умолчанию): ";
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "http://192.168.0.105:4747/video?500x500";
        }
        cout << "IP-камера: " << videoSource << endl;
    } else if (choice == 4) {
        cout << "Введите путь к видеофайлу (Enter = по умолчанию): ";
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/home/selbizo/CV/dataset/videos/PXL_1.mp4";
        }
        cout << "Видео: " << videoSource << endl;
    } else {
        cout << "Камера: " << videoSource << endl;
        cin.ignore();
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/home/pi/opencv_projects/videos/PXL_1.mp4";
        }
    }
    
    // Инициализация и запуск стабилизатора
    VideoStabilizer stabilizer;
    stabilizer.setUseFP16(useFP16);
    stabilizer.start(useVideo, imageFolderPath);
    
    cout << "\nУправление:" << endl;
    cout << "  ESC или Q - выход" << endl;
    cout << "  Пробел   - пауза" << endl;
    cout << "  F         - сохранить текущий кадр" << endl;
    cout << "  D         - переключить режим отладки" << endl;
    cout << "  S/W       - увеличить/уменьшить область кадра" << endl;
    cout << "  U/I       - увеличить/уменьшить коэффициент заполнения (D)" << endl;
    cout << "  J/K       - увеличить/уменьшить NSR" << endl;
    if (useFP16) {
        cout << "\n[GPU] FP16 оптимизация активирована" << endl;
    }
    
    try {
        while (stabilizer.running) {
            this_thread::sleep_for(chrono::seconds(1));
        }
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    stabilizer.stop();
    cout << "Program finished successfully" << endl;
    return 0;
}
