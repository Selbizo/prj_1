#include "gyro/sbgc_protocol.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <termios.h>
#include <unistd.h>

namespace sbgc {

namespace {

void putInt16LE(GyroPacket& p, size_t pos, int16_t value) {
    uint16_t u = static_cast<uint16_t>(value);
    p[pos] = static_cast<uint8_t>(u & 0xFF);
    p[pos + 1] = static_cast<uint8_t>((u >> 8) & 0xFF);
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

GyroPacket buildGyroCorrectionV1(const GyroCorrection& c) {
    GyroPacket p{};
    p[0] = 0x3E;
    p[1] = CMD_GYRO_CORRECTION;
    p[2] = 9;
    p[3] = 0x54;  // (75 + 9) & 0xFF
    p[4] = c.imuType;
    putInt16LE(p, 5, c.x);
    putInt16LE(p, 7, c.y);
    putInt16LE(p, 9, c.z);
    putInt16LE(p, 11, c.heading);

    uint8_t sum = 0;
    for (size_t i = 4; i < 13; ++i) sum = static_cast<uint8_t>(sum + p[i]);
    p[13] = sum;
    return p;
}

std::string toHex(const GyroPacket& packet) {
    static const char* hex = "0123456789ABCDEF";
    std::string s;
    s.reserve(packet.size() * 3);
    for (size_t i = 0; i < packet.size(); ++i) {
        if (i) s.push_back(' ');
        s.push_back(hex[(packet[i] >> 4) & 0xF]);
        s.push_back(hex[packet[i] & 0xF]);
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
        std::fprintf(stderr, "[GYRO] Неподдерживаемый baud rate: %d\n", baud);
        return false;
    }

    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
        std::fprintf(stderr, "[GYRO] Не удалось открыть %s: %s\n", port.c_str(), std::strerror(errno));
        return false;
    }

    termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
        std::fprintf(stderr, "[GYRO] tcgetattr(%s): %s\n", port.c_str(), std::strerror(errno));
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
        std::fprintf(stderr, "[GYRO] tcsetattr(%s): %s\n", port.c_str(), std::strerror(errno));
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

bool SbgcSerial::write(const GyroPacket& packet) {
    if (fd_ < 0) return false;
    size_t written = 0;
    while (written < packet.size()) {
        ssize_t n = ::write(fd_, packet.data() + written, packet.size() - written);
        if (n > 0) {
            written += static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            pollfd pfd{fd_, POLLOUT, 0};
            if (poll(&pfd, 1, 100) <= 0) return false;
            continue;
        }
        return false;
    }
    return true;
}

} // namespace sbgc
