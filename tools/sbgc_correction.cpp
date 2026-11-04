// Ручная отправка CMD_GYRO_CORRECTION (#75) для экспериментальной калибровки.
//
// Сборка (из корня проекта):
//   OC=/usr/local/opencv-4.12.0   # OpenCV не нужен, только -Isrc
//   g++ -O2 -std=gnu++17 -Isrc tools/sbgc_correction.cpp \
//       src/gyro/sbgc_protocol.cpp -o /tmp/sbgc_correction
//
// Примеры:
//   /tmp/sbgc_correction --self-test
//   /tmp/sbgc_correction --x 100 --y 0 --z 0 --dry-run
//   /tmp/sbgc_correction --port /dev/ttyUSB0 --send --x 100 --y 0 --z 0 --wait
//
// По умолчанию команда НЕ отправляется: нужен явный --send.

#include "gyro/sbgc_protocol.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std;

static void usage(const char* prog) {
    printf(
        "Использование: %s [опции]\n\n"
        "  --self-test          проверить CRC16 и сборку пакета\n"
        "  --port <dev>         последовательный порт (по умолчанию /dev/ttyUSB0)\n"
        "  --baud <n>           скорость (по умолчанию 115200)\n"
        "  --protocol <1|2>     версия протокола (по умолчанию 1)\n"
        "  --imu <0|1>          0 — main IMU, 1 — frame IMU\n"
        "  --x/--y/--z <int16>  GYRO_ZERO_CORR, единицы 0.001 gyro sensor unit\n"
        "  --heading <int16>    GYRO_ZERO_HEADING_CORR (обычно 0)\n"
        "  --send               реально отправить (без флага — dry-run)\n"
        "  --repeat <n>         отправить n раз (по умолчанию 1)\n"
        "  --interval <ms>      пауза между повторами (по умолчанию 100)\n"
        "  --wait               ждать CMD_CONFIRM (#67) до 500 мс\n",
        prog);
}

static bool selfTest() {
    const vector<uint8_t> example = {0x56, 0x02, 0x58, 0x00, 0x00};
    uint16_t crc = sbgc::crc16(example.data(), example.size());
    bool ok = (crc == 0x13E6);
    printf("CRC16 CMD_BOARD_INFO: %02X %02X (ожидается E6 13) — %s\n",
           crc & 0xFF, (crc >> 8) & 0xFF, ok ? "OK" : "FAIL");

    sbgc::GyroCorrection z;
    auto v1 = sbgc::buildGyroCorrectionV1(z);
    printf("v1 zero #75: %s\n", sbgc::toHex(v1).c_str());
    printf("  длина %zu, ожидается 14 байт; заголовок 3E 4B 09 54 — %s\n",
           v1.size(), (v1.size() == 14 && v1[0] == 0x3E && v1[1] == 75 &&
                       v1[2] == 9 && v1[3] == 0x54) ? "OK" : "FAIL");
    return ok;
}

int main(int argc, char** argv) {
    string port = "/dev/ttyUSB0";
    int baud = 115200;
    int protocol = 1;
    int imu = 0;
    int x = 0, y = 0, z = 0, heading = 0;
    int repeat = 1;
    int intervalMs = 100;
    bool send = false;
    bool wait = false;
    bool test = false;

    for (int i = 1; i < argc; ++i) {
        string a = argv[i];
        auto next = [&](int& out) {
            if (i + 1 >= argc) return false;
            out = atoi(argv[++i]);
            return true;
        };
        if (a == "--help" || a == "-h") { usage(argv[0]); return 0; }
        else if (a == "--self-test") test = true;
        else if (a == "--port" && i + 1 < argc) port = argv[++i];
        else if (a == "--baud" && next(baud)) {}
        else if (a == "--protocol" && next(protocol)) {}
        else if (a == "--imu" && next(imu)) {}
        else if (a == "--x" && next(x)) {}
        else if (a == "--y" && next(y)) {}
        else if (a == "--z" && next(z)) {}
        else if (a == "--heading" && next(heading)) {}
        else if (a == "--repeat" && next(repeat)) {}
        else if (a == "--interval" && next(intervalMs)) {}
        else if (a == "--send") send = true;
        else if (a == "--dry-run") send = false;
        else if (a == "--wait") wait = true;
        else { fprintf(stderr, "Неизвестный аргумент: %s\n", a.c_str()); usage(argv[0]); return 2; }
    }

    if (test) return selfTest() ? 0 : 1;

    sbgc::GyroCorrection c;
    c.imuType = static_cast<uint8_t>(imu ? 1 : 0);
    c.x = static_cast<int16_t>(x);
    c.y = static_cast<int16_t>(y);
    c.z = static_cast<int16_t>(z);
    c.heading = static_cast<int16_t>(heading);

    vector<uint8_t> packet = (protocol == 2)
        ? sbgc::buildGyroCorrectionV2(c)
        : sbgc::buildGyroCorrectionV1(c);

    printf("CMD_GYRO_CORRECTION #75: imu=%d x=%d y=%d z=%d heading=%d\n",
           imu, x, y, z, heading);
    printf("packet (protocol v%d, %zu bytes): %s\n",
           protocol, packet.size(), sbgc::toHex(packet).c_str());

    if (!send) {
        printf("DRY-RUN: добавьте --send для реальной записи в %s\n", port.c_str());
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
        if (wait) {
            vector<uint8_t> payload;
            int ver = 0;
            if (serial.readResponse(sbgc::CMD_CONFIRM, payload, 500, ver)) {
                printf("Ответ CMD_CONFIRM (%d байт, v%d): %s\n",
                       (int)payload.size(), ver, sbgc::toHex(payload).c_str());
            } else {
                printf("Ответ CMD_CONFIRM не получен за 500 мс\n");
            }
        }
        if (r + 1 < repeat) this_thread::sleep_for(chrono::milliseconds(intervalMs));
    }
    return 0;
}
