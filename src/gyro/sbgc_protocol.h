#pragma once

// Протокол SimpleBGC32 (BaseCam) для команды CMD_GYRO_CORRECTION (#75).
// Источник: SimpleBGC32 Serial API Protocol Specification, стр. 61.
//
// Формат кадра (protocol v1, стартовый байт 0x3E):
//   0x3E | CMD | SIZE | (CMD+SIZE)&0xFF | PAYLOAD... | SUM(PAYLOAD)&0xFF
//
// Формат кадра (protocol v2, стартовый байт 0x24, прошивка 2.68b0+):
//   0x24 | CMD | SIZE | (CMD+SIZE)&0xFF | PAYLOAD... | CRC16(header+payload, LE)
//
// CMD_GYRO_CORRECTION (#75), payload 9 байт:
//   IMU_TYPE             1u   0=main IMU, 1=frame IMU
//   GYRO_ZERO_CORR[3]    2s*3 int16 LE, порядок X, Y, Z
//   GYRO_ZERO_HEADING_CORR 2s int16 LE
//   Единицы: 0.001 "gyro sensor unit" (сырые LSB гироскопа).

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sbgc {

constexpr uint8_t START_V1 = 0x3E;
constexpr uint8_t START_V2 = 0x24;
constexpr uint8_t CMD_GYRO_CORRECTION = 75;
constexpr uint8_t CMD_CONFIRM = 67;
constexpr uint8_t CMD_ERROR = 255;

struct GyroCorrection {
    uint8_t imuType = 0;   // 0 — main IMU, 1 — frame IMU
    int16_t x = 0;
    int16_t y = 0;
    int16_t z = 0;
    int16_t heading = 0;   // GYRO_ZERO_HEADING_CORR, обычно 0
};

// CRC16 из Appendix A спецификации (полином 0x8005, начальное значение 0).
uint16_t crc16(const uint8_t* data, size_t length);

// Сборка кадра команды #75. Обе версии протокола.
std::vector<uint8_t> buildGyroCorrectionV1(const GyroCorrection& c);
std::vector<uint8_t> buildGyroCorrectionV2(const GyroCorrection& c);

// Проверка/разбор входящего кадра. Возвращает true, если найден валидный кадр.
// Поддерживает оба стартовых байта; protocolVersion = 1 или 2.
bool parseFrame(const std::vector<uint8_t>& buf, uint8_t& cmd,
                std::vector<uint8_t>& payload, int& protocolVersion);

std::string toHex(const std::vector<uint8_t>& data);

// Простой последовательный порт 8N1 без лишних зависимостей.
class SbgcSerial {
public:
    SbgcSerial() = default;
    ~SbgcSerial();
    SbgcSerial(const SbgcSerial&) = delete;
    SbgcSerial& operator=(const SbgcSerial&) = delete;

    bool open(const std::string& port, int baud);
    void close();
    bool isOpen() const { return fd_ >= 0; }
    bool write(const std::vector<uint8_t>& data);

    // Ждёт ответ с заданным CMD_ID (для #75 обычно CMD_CONFIRM=67).
    // timeoutMs = 0 — только прочитать уже накопленное.
    bool readResponse(uint8_t expectedCmd, std::vector<uint8_t>& payload,
                      int timeoutMs, int& protocolVersion);

private:
    int fd_ = -1;
};

} // namespace sbgc
