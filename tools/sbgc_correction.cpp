// Ручная отправка CMD_GYRO_CORRECTION (#75) для проверки осей и знаков.
// Без --send работает как dry-run.
//
// Сборка (из корня проекта):
//   g++ -O2 -std=gnu++17 -Isrc tools/sbgc_correction.cpp
//       src/gyro/sbgc_protocol.cpp -o /tmp/sbgc_correction

#include "gyro/sbgc_protocol.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace std;

static void usage(const char* prog) {
    printf(
        "Использование: %s [опции]\n\n"
        "  --port <dev>         порт (по умолчанию /dev/ttyUSB0)\n"
        "  --baud <n>           скорость (по умолчанию 115200)\n"
        "  --imu <0|1>          0 — main IMU, 1 — frame IMU\n"
        "  --x/--y/--z <int16>  поправки X/Y/Z, единицы 0.001 gyro LSB\n"
        "  --heading <int16>    GYRO_ZERO_HEADING_CORR (обычно 0)\n"
        "  --send               реально отправить (без флага — dry-run)\n"
        "  --repeat <n>         отправить n раз (1)\n"
        "  --interval <ms>      пауза между повторами (100)\n"
        "  --self-test          проверить формирование пакета\n",
        prog);
}

int main(int argc, char** argv) {
    string port = "/dev/ttyUSB0";
    int baud = 115200;
    int imu = 0, x = 0, y = 0, z = 0, heading = 0;
    int repeat = 1, intervalMs = 100;
    bool send = false, test = false;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto next = [&](int& out) {
            if (i + 1 >= argc) return false;
            out = atoi(argv[++i]);
            return true;
        };
        if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
        else if (a == "--port" && i + 1 < argc) port = argv[++i];
        else if (a == "--baud" && next(baud)) {}
        else if (a == "--imu" && next(imu)) {}
        else if (a == "--x" && next(x)) {}
        else if (a == "--y" && next(y)) {}
        else if (a == "--z" && next(z)) {}
        else if (a == "--heading" && next(heading)) {}
        else if (a == "--repeat" && next(repeat)) {}
        else if (a == "--interval" && next(intervalMs)) {}
        else if (a == "--send") send = true;
        else if (a == "--dry-run") send = false;
        else if (a == "--self-test") test = true;
        else { fprintf(stderr, "Неизвестный аргумент: %s\n", a.c_str()); return 2; }
    }

    sbgc::GyroCorrection c;
    c.imuType = static_cast<uint8_t>(imu ? 1 : 0);
    c.x = static_cast<int16_t>(x);
    c.y = static_cast<int16_t>(y);
    c.z = static_cast<int16_t>(z);
    c.heading = static_cast<int16_t>(heading);

    sbgc::GyroPacket packet = sbgc::buildGyroCorrectionV1(c);
    printf("CMD_GYRO_CORRECTION #75: imu=%d x=%d y=%d z=%d heading=%d\n",
           imu, x, y, z, heading);
    printf("packet: %s\n", sbgc::toHex(packet).c_str());

    if (test) {
        bool ok = packet.size() == 14 && packet[0] == 0x3E && packet[1] == 75 &&
                  packet[2] == 9 && packet[3] == 0x54;
        printf("self-test: %s\n", ok ? "OK" : "FAIL");
        return ok ? 0 : 1;
    }

    if (!send) {
        printf("DRY-RUN: добавьте --send для записи в %s\n", port.c_str());
        return 0;
    }

    sbgc::SbgcSerial serial;
    if (!serial.open(port, baud)) return 1;
    for (int r = 0; r < repeat; ++r) {
        if (!serial.write(packet)) {
            fprintf(stderr, "Ошибка записи в %s\n", port.c_str());
            return 1;
        }
        printf("Отправлено %d/%d\n", r + 1, repeat);
        if (r + 1 < repeat) this_thread::sleep_for(chrono::milliseconds(intervalMs));
    }
    return 0;
}
