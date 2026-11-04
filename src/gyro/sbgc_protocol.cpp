#include "gyro/sbgc_protocol.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>
#include <utility>

namespace sbgc {

namespace {

void putInt16LE(std::vector<uint8_t>& v, int16_t value) {
    uint16_t u = static_cast<uint16_t>(value);
    v.push_back(static_cast<uint8_t>(u & 0xFF));
    v.push_back(static_cast<uint8_t>((u >> 8) & 0xFF));
}

int16_t getInt16LE(const std::vector<uint8_t>& v, size_t pos) {
    uint16_t u = static_cast<uint16_t>(v[pos]) |
                 (static_cast<uint16_t>(v[pos + 1]) << 8);
    return static_cast<int16_t>(u);
}

speed_t toSpeed(int baud) {
    switch (baud) {
        case 9600: return B9600;
        case 19200: return B19200;
        case 38400: return B38400;
        case 57600: return B57600;
        case 115200: return B115200;
        case 230400: return B230400;
        default: return 0;
    }
}

} // namespace

uint16_t crc16(const uint8_t* data, size_t length) {
    uint16_t crc = 0;
    for (size_t i = 0; i < length; ++i) {
        for (int bit = 0; bit < 8; ++bit) {
            uint8_t dataBit = (data[i] >> bit) & 1;
            uint8_t crcBit = (crc >> 15) & 1;
            crc <<= 1;
            if (dataBit != crcBit) crc ^= 0x8005;
        }
    }
    return crc;
}

std::vector<uint8_t> buildGyroCorrectionV1(const GyroCorrection& c) {
    std::vector<uint8_t> payload;
    payload.reserve(9);
    payload.push_back(c.imuType);
    putInt16LE(payload, c.x);
    putInt16LE(payload, c.y);
    putInt16LE(payload, c.z);
    putInt16LE(payload, c.heading);

    std::vector<uint8_t> frame;
    frame.reserve(4 + payload.size() + 1);
    frame.push_back(START_V1);
    frame.push_back(CMD_GYRO_CORRECTION);
    frame.push_back(static_cast<uint8_t>(payload.size()));
    frame.push_back(static_cast<uint8_t>((CMD_GYRO_CORRECTION + payload.size()) & 0xFF));
    frame.insert(frame.end(), payload.begin(), payload.end());
    uint8_t sum = 0;
    for (uint8_t b : payload) sum = static_cast<uint8_t>(sum + b);
    frame.push_back(sum);
    return frame;
}

std::vector<uint8_t> buildGyroCorrectionV2(const GyroCorrection& c) {
    std::vector<uint8_t> payload;
    payload.reserve(9);
    payload.push_back(c.imuType);
    putInt16LE(payload, c.x);
    putInt16LE(payload, c.y);
    putInt16LE(payload, c.z);
    putInt16LE(payload, c.heading);

    std::vector<uint8_t> frame;
    frame.reserve(4 + payload.size() + 2);
    frame.push_back(START_V2);
    frame.push_back(CMD_GYRO_CORRECTION);
    frame.push_back(static_cast<uint8_t>(payload.size()));
    frame.push_back(static_cast<uint8_t>((CMD_GYRO_CORRECTION + payload.size()) & 0xFF));
    frame.insert(frame.end(), payload.begin(), payload.end());
    uint16_t crc = crc16(frame.data() + 1, frame.size() - 1);
    frame.push_back(static_cast<uint8_t>(crc & 0xFF));
    frame.push_back(static_cast<uint8_t>((crc >> 8) & 0xFF));
    return frame;
}

bool parseFrame(const std::vector<uint8_t>& buf, uint8_t& cmd,
                std::vector<uint8_t>& payload, int& protocolVersion) {
    for (size_t start = 0; start + 4 <= buf.size(); ++start) {
        uint8_t first = buf[start];
        if (first != START_V1 && first != START_V2) continue;

        cmd = buf[start + 1];
        uint8_t size = buf[start + 2];
        uint8_t hdrCheck = buf[start + 3];
        if (hdrCheck != static_cast<uint8_t>((cmd + size) & 0xFF)) continue;

        if (first == START_V1) {
            size_t total = 4 + static_cast<size_t>(size) + 1;
            if (start + total > buf.size()) continue;
            uint8_t sum = 0;
            for (size_t i = 0; i < size; ++i) sum = static_cast<uint8_t>(sum + buf[start + 4 + i]);
            if (sum != buf[start + 4 + size]) continue;
            payload.assign(buf.begin() + start + 4, buf.begin() + start + 4 + size);
            protocolVersion = 1;
            return true;
        }

        size_t total = 4 + static_cast<size_t>(size) + 2;
        if (start + total > buf.size()) continue;
        uint16_t got = static_cast<uint16_t>(buf[start + 4 + size]) |
                       (static_cast<uint16_t>(buf[start + 4 + size + 1]) << 8);
        uint16_t want = crc16(buf.data() + start + 1, 3 + static_cast<size_t>(size));
        if (got != want) continue;
        payload.assign(buf.begin() + start + 4, buf.begin() + start + 4 + size);
        protocolVersion = 2;
        return true;
    }
    return false;
}

std::string toHex(const std::vector<uint8_t>& data) {
    static const char* hex = "0123456789ABCDEF";
    std::string s;
    s.reserve(data.size() * 3);
    for (size_t i = 0; i < data.size(); ++i) {
        if (i) s.push_back(' ');
        s.push_back(hex[(data[i] >> 4) & 0xF]);
        s.push_back(hex[data[i] & 0xF]);
    }
    return s;
}

SbgcSerial::~SbgcSerial() {
    close();
}

bool SbgcSerial::open(const std::string& port, int baud) {
    close();
    speed_t speed = toSpeed(baud);
    if (speed == 0) {
        std::fprintf(stderr, "[SBGC] Неподдерживаемый baud rate: %d\n", baud);
        return false;
    }

    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::fprintf(stderr, "[SBGC] Не удалось открыть %s: %s\n", port.c_str(), std::strerror(errno));
        return false;
    }

    termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
        std::fprintf(stderr, "[SBGC] tcgetattr(%s): %s\n", port.c_str(), std::strerror(errno));
        close();
        return false;
    }
    cfmakeraw(&tio);
    tio.c_cflag |= CLOCAL | CREAD;
    tio.c_cflag &= ~CSTOPB;
    tio.c_cflag &= ~PARENB;
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;
    tio.c_cc[VMIN] = 0;
    tio.c_cc[VTIME] = 0;
    cfsetispeed(&tio, speed);
    cfsetospeed(&tio, speed);
    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
        std::fprintf(stderr, "[SBGC] tcsetattr(%s): %s\n", port.c_str(), std::strerror(errno));
        close();
        return false;
    }
    tcflush(fd_, TCIOFLUSH);
    return true;
}

void SbgcSerial::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

bool SbgcSerial::write(const std::vector<uint8_t>& data) {
    if (fd_ < 0) return false;
    size_t written = 0;
    while (written < data.size()) {
        ssize_t n = ::write(fd_, data.data() + written, data.size() - written);
        if (n > 0) {
            written += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd pfd{fd_, POLLOUT, 0};
            int rc = poll(&pfd, 1, 100);
            if (rc <= 0) return false;
            continue;
        }
        return false;
    }
    tcdrain(fd_);
    return true;
}

bool SbgcSerial::readResponse(uint8_t expectedCmd, std::vector<uint8_t>& payload,
                              int timeoutMs, int& protocolVersion) {
    if (fd_ < 0) return false;
    std::vector<uint8_t> buf;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

    while (true) {
        uint8_t cmd = 0;
        std::vector<uint8_t> parsed;
        int ver = 0;
        if (parseFrame(buf, cmd, parsed, ver)) {
            if (cmd == expectedCmd) {
                payload = std::move(parsed);
                protocolVersion = ver;
                return true;
            }
            // Чужой валидный кадр: сдвигаем буфер на байт и ищем дальше.
            buf.erase(buf.begin());
            continue;
        }

        int remaining = static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
            deadline - std::chrono::steady_clock::now()).count());
        if (remaining < 0) remaining = 0;

        pollfd pfd{fd_, POLLIN, 0};
        int rc = poll(&pfd, 1, remaining);
        if (rc <= 0) return false;

        uint8_t tmp[256];
        ssize_t n = ::read(fd_, tmp, sizeof(tmp));
        if (n <= 0) return false;
        buf.insert(buf.end(), tmp, tmp + n);

        // Защита от мусора: не даём буферу расти бесконечно.
        if (buf.size() > 4096) buf.erase(buf.begin(), buf.end() - 256);
    }
}

} // namespace sbgc
