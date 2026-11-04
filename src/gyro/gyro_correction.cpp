#include "gyro/gyro_correction.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

int16_t toInt16(double v) {
    v = std::max(-32768.0, std::min(32767.0, v));
    return static_cast<int16_t>(std::lround(v));
}

void assign(sbgc::GyroCorrection& c, int axis, int16_t value) {
    if (axis == 0) c.x = value;
    else if (axis == 1) c.y = value;
    else if (axis == 2) c.z = value;
}

} // namespace

void GyroCorrection::configure(const GyroConfig& cfg, int frameWidth, int frameHeight) {
    cfg_ = cfg;
    (void)frameHeight;
    double fov = clampd(cfg_.fovHdeg, 5.0, 170.0) * 3.14159265358979323846 / 180.0;
    fx_ = (frameWidth > 0) ? (frameWidth / 2.0) / std::tan(fov / 2.0) : 1.0;
    fy_ = fx_;
    if (fx_ < 1.0) fx_ = 1.0;
    if (fy_ < 1.0) fy_ = 1.0;
    telem_ = GyroTelemetry{};
    configured_ = true;
}

void GyroCorrection::update(double dxPixels, double dyPixels, double daRadians,
                            double dtSec, int points) {
    if (!configured_ || dtSec <= 1e-4 || dtSec > 0.5 || points < 6) return;

    const double yawInst = std::atan(dxPixels / fx_) * kRadToDeg / dtSec;
    const double pitchInst = std::atan(dyPixels / fy_) * kRadToDeg / dtSec;
    const double rollInst = daRadians * kRadToDeg / dtSec;

    const double alpha = 1.0 - std::exp(-dtSec / std::max(0.05, cfg_.tauSec));
    telem_.yawRateDps += alpha * (yawInst - telem_.yawRateDps);
    telem_.pitchRateDps += alpha * (pitchInst - telem_.pitchRateDps);
    telem_.rollRateDps += alpha * (rollInst - telem_.rollRateDps);

    auto integrate = [&](double rate, double& cmd) {
        if (std::abs(rate) < cfg_.deadbandDps) return;
        if (std::abs(rate) > cfg_.maxRateDps) return;  // это движение, не дрейф
        cmd -= cfg_.integralGain * rate * dtSec;
        cmd = clampd(cmd, -cfg_.maxDps, cfg_.maxDps);
    };
    integrate(telem_.yawRateDps, telem_.yawCmdDps);
    integrate(telem_.pitchRateDps, telem_.pitchCmdDps);
    integrate(telem_.rollRateDps, telem_.rollCmdDps);
}

sbgc::GyroCorrection GyroCorrection::packet() const {
    sbgc::GyroCorrection c;
    c.imuType = 0;
    assign(c, cfg_.pitchAxis, toInt16(telem_.pitchCmdDps * cfg_.scaleUnitsPerDps * cfg_.pitchSign));
    assign(c, cfg_.yawAxis, toInt16(telem_.yawCmdDps * cfg_.scaleUnitsPerDps * cfg_.yawSign));
    assign(c, cfg_.rollAxis, toInt16(telem_.rollCmdDps * cfg_.scaleUnitsPerDps * cfg_.rollSign));
    c.heading = 0;
    return c;
}
