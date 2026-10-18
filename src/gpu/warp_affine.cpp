#include "config/structures.h"
#include <opencv2/core/ocl.hpp>
#include <iostream>
#include <exception>

using namespace cv;
using namespace std;

// ========================= ДЕТЕКЦИЯ GPU =========================
GPUCapabilities detectGPUCapabilities() {
    GPUCapabilities caps;
    
    if (!ocl::haveOpenCL()) {
        cerr << "[GPU] OpenCL не доступен" << endl;
        return caps;
    }
    
    ocl::Context context = ocl::Context::getDefault();
    if (!context.ptr()) {
        cerr << "[GPU] Не удалось получить OpenCL контекст" << endl;
        return caps;
    }
    
    const ocl::Device& device = ocl::Device::getDefault();
    caps.gpuName = device.name();
    cout << "[GPU] Найдено устройство: " << caps.gpuName << endl;
    
    string extensions = device.extensions();
    caps.supportsFP16 = (extensions.find("cl_khr_fp16") != string::npos);
    
    if (caps.supportsFP16) {
        cout << "[GPU] ✓ FP16 поддерживается (cl_khr_fp16)" << endl;
    } else {
        cout << "[GPU] ✗ FP16 расширение не найдено" << endl;
    }
    
    if (caps.gpuName.find("Mali") != string::npos) {
        caps.supportsHalfType = true;
        // Заявленная поддержка half != рабочая: ядро OpenCV для half3 на Mali-G52
        // не собирается (CL_BUILD_PROGRAM_FAILURE), поэтому FP16 не используется.
        cout << "[GPU] ✓ Обнаружена Mali GPU (half заявлен драйвером, но не используется)" << endl;
    }
    
    return caps;
}

// ========================= WARP AFFINE =========================
void warpAffineOptimized(InputArray src, OutputArray dst, InputArray M, Size dsize,
                         int flags, int borderMode, const Scalar& borderValue, bool useFP16) {
    if (!USE_FP16_WARP || !useFP16) {
        warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
        return;
    }
    
    try {
        UMat src_umat = src.getUMat();
        UMat src_fp16, result_fp16;
        
        if (src.type() == CV_8UC3) {
            UMat src_fp32;
            src_umat.convertTo(src_fp32, CV_32F, 1.0 / 255.0);
            src_fp32.convertTo(src_fp16, CV_16F);
        } else {
            src_umat.convertTo(src_fp16, CV_16F);
        }
        
        warpAffine(src_fp16, result_fp16, M, dsize, flags, borderMode);
        
        if (dst.type() == CV_8UC3 || dst.getMat().type() == CV_8UC3) {
            UMat result_fp32;
            result_fp16.convertTo(result_fp32, CV_32F);
            result_fp32.convertTo(dst, CV_8U, 255.0);
        } else {
            result_fp16.convertTo(dst, CV_32F);
        }
    } catch (const exception& e) {
        cerr << "[GPU] Ошибка FP16 warpAffine, используется стандартная версия: " << e.what() << endl;
        warpAffine(src, dst, M, dsize, flags, borderMode, borderValue);
    }
}
