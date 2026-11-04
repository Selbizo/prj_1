#pragma once

// Минимальный протокол SimpleBGC для CMD_GYRO_CORRECTION (#75).
// Кадр protocol v1: 0x3E | 75 | 9 | 0x54 | payload[9] | SUM(payload).

#include <array>
#include <cstdint>
#include <string>

namespace sbgc {

constexpr uint8_t CMD_GYRO_CORRECTION = 75;

struct GyroCorrection {
    uint8_t imuType = 0;   // 0 — main IMU, 1 — frame IMU
    int16_t x = 0;
    int16_t y = 0;
    int16_t z = 0;
    int16_t heading = 0;   // GYRO_ZERO_HEADING_CORR, обычно 0
};

using GyroPacket = std::array<uint8_t, 14>;

GyroPacket buildGyroCorrectionV1(const GyroCorrection& c);
std::string toHex(const GyroPacket& packet);

// Простой UART 8N1. Без чтения ответов и CRC — только отправка #75.
class SbgcSerial {
public:
    SbgcSerial() = default;
    ~SbgcSerial();
    SbgcSerial(const SbgcSerial&) = delete;
    SbgcSerial& operator=(const SbgcSerial&) = delete;

    bool open(const std::string& port, int baud);
    void close();
    bool isOpen() const { return fd_ >= 0; }
    bool write(const GyroPacket& packet);

private:
    int fd_ = -1;
};

} // namespace sbgc
