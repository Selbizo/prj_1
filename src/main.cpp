#include "config/config.h"
#include "config/structures.h"
#include "stabilizer/video_stabilizer.h"
#include "gpu/warp_affine.h"
#include "utils/helpers.h"

#include <iostream>
#include <iomanip>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>

using namespace std;

// ========================= АРГУМЕНТЫ КОМАНДНОЙ СТРОКИ =========================
static bool startsWith(const string& s, const string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
} // end startsWith

static bool axisIndex(const string& s, int& out) {
    if (s == "x" || s == "X" || s == "0") { out = 0; return true; }
    if (s == "y" || s == "Y" || s == "1") { out = 1; return true; }
    if (s == "z" || s == "Z" || s == "2") { out = 2; return true; }
    return false;
} // end axisIndex

static void printUsage(const char* prog) {
    cout << "Использование: " << prog << " [опции]\n\n"
         << "  --source <spec>  MIPI-камера V4L2:\n"
         << "                     camera                — /dev/video0\n"
         << "                     /dev/videoN           — конкретный узел\n"
         << "  --estimate-only  только вычисление смещения (максимальный fps):\n"
         << "                     два потока (захват + трекинг), без стабилизации\n"
         << "                     и отображения; камера читается сырым YUYV.\n"
         << "                     Каждую секунду печатается строка [EST] с dx/dy/da.\n"
         << "  --duration <sec>  работать заданное число секунд и корректно выйти\n"
         << "  --no-stab        стартовать без контура стабилизации (пиковый fps)\n"
         << "  --max-fps-raw    обычный режим с окном, но камера читается сырым YUYV\n"
         << "                     (одна конверсия YUYV->BGR для окна). С --no-stab\n"
         << "                     даёт ~100-115 fps; в --estimate-only игнорируется.\n"
         << "  --no-gui         без окна (в консоль идут только строки [PERF]/[EST])\n"
         << "  --cam-auto       не отключать автоэкспозицию/автобаланс белого\n"
         << "                     (по умолчанию они выключаются ради тока)\n"
         << "  --drift-report   печатать оценку дрейфа (yaw/pitch/roll, deg/s, conf)\n"
         << "  --gyro-correction включить расчёт поправки CMD_GYRO_CORRECTION (#75);\n"
         << "                     без --gyro-enable работает dry-run (только лог)\n"
         << "  --gyro-port <dev> последовательный порт (по умолчанию /dev/ttyUSB0)\n"
         << "  --gyro-baud <n>   скорость порта (по умолчанию 115200)\n"
         << "  --gyro-enable     РАЗРЕШИТЬ реальную отправку команды #75\n"
         << "  --gyro-fov-h <deg> горизонтальный FOV камеры (по умолчанию 60)\n"
         << "  --gyro-scale <u>  единиц #75 на 1 град/с (по умолчанию 16400)\n"
         << "  --gyro-max <dps>  ограничение поправки, град/с (по умолчанию 2.0)\n"
         << "  --gyro-tau <sec>  постоянная времени ФНЧ дрейфа (по умолчанию 2.0)\n"
         << "  --gyro-rate <hz>  частота отправки #75 (по умолчанию 20)\n"
         << "  --gyro-axis-pitch|yaw|roll <x|y|z>  соответствие осей #75\n"
         << "  --gyro-sign-pitch|yaw|roll <+1|-1>  знаки осей #75\n"
         << "  -h, --help       эта справка\n\n"
         << "Без --source запрашивается путь к MIPI-камере (Enter = /dev/video0).\n"
         << "Переменные окружения: VS_ESTIMATE_ONLY, VS_MAX_FPS_RAW, VS_NO_STAB,\n"
         << "  VS_NO_GUI, VS_FAST_RESIZE, VS_OCV_THREADS, VS_PINNED, VS_WARP_CPU,\n"
         << "  VS_VERBOSE, VS_EST_VERBOSE, VS_DRIFT_REPORT, VS_GYRO_CORRECTION,\n"
         << "  VS_GYRO_PORT, VS_GYRO_BAUD, VS_GYRO_ENABLE, VS_GYRO_FOV_H,\n"
         << "  VS_GYRO_SCALE, VS_GYRO_MAX_DPS, VS_GYRO_TAU, VS_GYRO_RATE,\n"
         << "  VS_GYRO_MIN_CONF, VS_GYRO_DEADBAND, VS_GYRO_AXIS_PITCH/YAW/ROLL,\n"
         << "  VS_GYRO_SIGN_PITCH/YAW/ROLL, VS_GYRO_PROTOCOL, VS_CAM_AUTO.\n";
} // end printUsage

// ========================= ГЛАВНАЯ ФУНКЦИЯ =========================

int main(int argc, char** argv) {
    cout << "========================================" << endl;
    cout << " MULTI-THREADED VIDEO STABILIZER OPENCL " << endl;
    cout << "========================================" << endl;
    
    cv::ocl::setUseOpenCL(USE_OPENCL);
    int cores = getAvailableCores();
    cout << "CPU cores available: " << cores << endl;
    
    // Разбор аргументов. --no-stab / --no-gui выставляют те же переменные среды,
    // которые читает VideoStabilizer, поэтому поведение совпадает с env-вариантом.
    string sourceArg;
    double runSeconds = 0.0;
    bool estimateOnlyArg = false;
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--source" && i + 1 < argc) {
            sourceArg = argv[++i];
        } else if (arg == "--duration" && i + 1 < argc) {
            runSeconds = atof(argv[++i]);
        } else if (arg == "--estimate-only") {
            estimateOnlyArg = true;
            setenv("VS_ESTIMATE_ONLY", "1", 1);
            setenv("VS_NO_GUI", "1", 1);
        } else if (arg == "--no-stab") {
            setenv("VS_NO_STAB", "1", 1);
        } else if (arg == "--max-fps-raw") {
            // Обычный режим с окном, но без внутренней конверсии OpenCV:
            // сырой YUYV + одна YUYV->BGR для окна. Максимальный эффект с --no-stab.
            setenv("VS_MAX_FPS_RAW", "1", 1);
        } else if (arg == "--no-gui") {
            setenv("VS_NO_GUI", "1", 1);
        } else if (arg == "--cam-auto") {
            setenv("VS_CAM_AUTO", "1", 1);
        } else if (arg == "--drift-report") {
            setenv("VS_DRIFT_REPORT", "1", 1);
        } else if (arg == "--gyro-correction") {
            setenv("VS_GYRO_CORRECTION", "1", 1);
        } else if (arg == "--gyro-enable") {
            setenv("VS_GYRO_ENABLE", "1", 1);
        } else if (arg == "--gyro-port" && i + 1 < argc) {
            setenv("VS_GYRO_PORT", argv[++i], 1);
        } else if (arg == "--gyro-baud" && i + 1 < argc) {
            setenv("VS_GYRO_BAUD", argv[++i], 1);
        } else if (arg == "--gyro-fov-h" && i + 1 < argc) {
            setenv("VS_GYRO_FOV_H", argv[++i], 1);
        } else if (arg == "--gyro-scale" && i + 1 < argc) {
            setenv("VS_GYRO_SCALE", argv[++i], 1);
        } else if (arg == "--gyro-max" && i + 1 < argc) {
            setenv("VS_GYRO_MAX_DPS", argv[++i], 1);
        } else if (arg == "--gyro-tau" && i + 1 < argc) {
            setenv("VS_GYRO_TAU", argv[++i], 1);
        } else if (arg == "--gyro-rate" && i + 1 < argc) {
            setenv("VS_GYRO_RATE", argv[++i], 1);
        } else if ((arg == "--gyro-axis-pitch" || arg == "--gyro-axis-yaw" ||
                    arg == "--gyro-axis-roll") && i + 1 < argc) {
            int axis = 0;
            if (!axisIndex(argv[++i], axis)) {
                cerr << "Ось должна быть x, y или z: " << arg << endl;
                return 2;
            }
            const char* env = (arg == "--gyro-axis-pitch") ? "VS_GYRO_AXIS_PITCH"
                            : (arg == "--gyro-axis-yaw")   ? "VS_GYRO_AXIS_YAW"
                                                           : "VS_GYRO_AXIS_ROLL";
            string v = to_string(axis);
            setenv(env, v.c_str(), 1);
        } else if ((arg == "--gyro-sign-pitch" || arg == "--gyro-sign-yaw" ||
                    arg == "--gyro-sign-roll") && i + 1 < argc) {
            int sign = atoi(argv[++i]) < 0 ? -1 : 1;
            const char* env = (arg == "--gyro-sign-pitch") ? "VS_GYRO_SIGN_PITCH"
                            : (arg == "--gyro-sign-yaw")   ? "VS_GYRO_SIGN_YAW"
                                                           : "VS_GYRO_SIGN_ROLL";
            string v = to_string(sign);
            setenv(env, v.c_str(), 1);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            cerr << "Неизвестный аргумент: " << arg << endl;
            printUsage(argv[0]);
            return 2;
        }
    }
    
    // Обнаружить возможности GPU. В режиме «только оценка смещения» GPU не нужен
    // (нет warpAffine), поэтому инициализацию OpenCL пропускаем — она заметно
    // тормозит старт и может тянуть фоновые потоки драйвера.
    bool useFP16 = false;
    if (!estimateOnlyArg) {
        cout << "\n[ИНИЦИАЛИЗАЦИЯ GPU]" << endl;
        GPUCapabilities gpuCaps = detectGPUCapabilities();
        useFP16 = gpuCaps.supportsFP16 || gpuCaps.supportsHalfType;
        if (!useFP16) {
            cout << "[GPU] Предупреждение: FP16 недоступен, используется стандартная обработка" << endl;
        }
        cout << endl;
    } else {
        cout << "\n[EST] GPU не используется: стабилизация и отображение отключены" << endl;
    }
    
    if (!sourceArg.empty()) {
        // Источник задан аргументом: поддерживается только MIPI/V4L2 камера.
        if (sourceArg == "camera" || sourceArg == "cam") {
            videoSource = "/dev/video0";
        } else if (startsWith(sourceArg, "/dev/video")) {
            videoSource = sourceArg;
        } else {
            cerr << "Источник не поддерживается: " << sourceArg << endl;
            cerr << "Доступны только MIPI-камера: camera | /dev/videoN" << endl;
            return 2;
        }
        cout << "Источник: MIPI-камера " << videoSource << endl;
    } else {
        // Интерактивное меню: единственный источник — MIPI-камера OS08A10.
        cout << "MIPI-камера (Enter = /dev/video0): ";
        getline(cin, videoSource);
        if (videoSource.empty()) {
            videoSource = "/dev/video0";
        }
        cout << "Камера: " << videoSource << endl;
    } // end else: интерактивный выбор камеры

    // Инициализация и запуск стабилизатора
    VideoStabilizer stabilizer;
    stabilizer.setUseFP16(useFP16);
    stabilizer.start();
    
    if (estimateOnlyArg) {
        cout << "\n[EST] Режим «только оценка смещения». Каждую секунду печатается:"
             << "\n      fps | dx, dy (пиксели полного кадра) | da (радианы и градусы)"
             << "\n      pts | det+trk | cap | rawQ | skip"
             << "\n      VS_EST_VERBOSE=1 — печатать смещение на каждом кадре"
             << (runSeconds > 0 ? "" : "\n      Остановка: Ctrl-C (или задайте --duration <сек>)") << endl;
    } else {
        cout << "\nУправление:" << endl;
        cout << "  ESC или Q - выход" << endl;
        cout << "  Пробел   - пауза" << endl;
        cout << "  F         - сохранить текущий кадр" << endl;
        cout << "  D         - переключить режим отладки" << endl;
        cout << "  B         - вкл/выкл контур стабилизации (пиковый fps)" << endl;
        cout << "  S/W       - увеличить/уменьшить область кадра" << endl;
        cout << "  U/I       - увеличить/уменьшить коэффициент заполнения (D)" << endl;
        cout << "  J/K       - увеличить/уменьшить NSR" << endl;
        if (useFP16 && USE_FP16_WARP) {
            cout << "\n[GPU] FP16 оптимизация активирована" << endl;
        } else if (useFP16) {
            cout << "\n[GPU] FP16 недоступен на этой платформе (ядро half3 не собирается "
                 << "OpenCL-компилятором Mali) — работает обычный warpAffine" << endl;
        }
    } // end else: обычный (интерактивный) режим
    
    // Время фактической работы конвейера — знаменатель для end-to-end fps в main.
    auto runStart = chrono::steady_clock::now();
    try {
        if (runSeconds > 0.0) {
            // Детерминированный прогон для замеров: ждём заданное время и гасим конвейер.
            auto deadline = chrono::steady_clock::now() + chrono::milliseconds(static_cast<long>(runSeconds * 1000.0));
            while (stabilizer.running && chrono::steady_clock::now() < deadline) {
                this_thread::sleep_for(chrono::milliseconds(100));
            } // end while (running && now < deadline)
            if (stabilizer.running) {
                cout << "\n[RUN] Задано " << runSeconds << " с — останавливаемся" << endl;
                stabilizer.running = false;
            } // end if (running)
        } else {
            while (stabilizer.running) {
                this_thread::sleep_for(chrono::seconds(1));
            } // end while (running)
        } // end else: без ограничения по времени
    } catch (...) {
        cout << "Main thread interrupted" << endl;
    }
    
    auto runEnd = chrono::steady_clock::now();
    stabilizer.stop();
    if (estimateOnlyArg) {
        double runSecs = chrono::duration<double>(runEnd - runStart).count();
        long long n = stabilizer.estimateFramesTotal.load();
        double fps = (runSecs > 0.0) ? static_cast<double>(n) / runSecs : 0.0;
        cout << "[EST] ИТОГО (end-to-end): " << n << " кадров за "
             << fixed << setprecision(2) << runSecs << " с = "
             << setprecision(2) << fps << " fps" << endl;
    }
    cout << "Program finished successfully" << endl;
    return 0;
}
