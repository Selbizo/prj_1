#include "gyro/gyro_correction.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace {
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;
constexpr double kPi = 3.14159265358979323846;
}

void GyroCorrection::configure(const GyroConfig& cfg, int frameWidth, int frameHeight) {
    cfg_ = cfg;
    (void)frameHeight;
    double fov = clampd(cfg_.fovHdeg, 5.0, 170.0) * kPi / 180.0;
    fx_ = (frameWidth > 0) ? (frameWidth / 2.0) / std::tan(fov / 2.0) : 1.0;
    fy_ = fx_;
    if (fx_ < 1.0) fx_ = 1.0;
    if (fy_ < 1.0) fy_ = 1.0;

    winYaw_.clear();
    winPitch_.clear();
    winRoll_.clear();
    yawFiltered_ = pitchFiltered_ = rollFiltered_ = 0.0;
    timeSinceValid_ = 0.0;
    telem_ = GyroTelemetry{};
    configured_ = true;
}

double GyroCorrection::clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

double GyroCorrection::median(std::deque<double> window) {
    if (window.empty()) return 0.0;
    std::vector<double> v(window.begin(), window.end());
    size_t mid = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + mid, v.end());
    double m = v[mid];
    if ((v.size() & 1u) == 0u && mid > 0) {
        double lower = *std::max_element(v.begin(), v.begin() + mid);
        m = 0.5 * (m + lower);
    }
    return m;
}

void GyroCorrection::pushWindow(std::deque<double>& window, double value) const {
    int cap = std::max(1, cfg_.windowSamples);
    window.push_back(value);
    while (static_cast<int>(window.size()) > cap) window.pop_front();
}

void GyroCorrection::update(double dxPixels, double dyPixels, double daRadians,
                            double dtSec, int points, double quality) {
    if (!configured_ || dtSec <= 1e-4 || dtSec > 0.5) {
        telem_.valid = false;
        return;
    }

    quality = clampd(quality, 0.0, 1.0);
    const bool trackOk = points >= cfg_.minPoints && quality >= cfg_.minQuality;

    const double yawInst = std::atan(dxPixels / fx_) * kRadToDeg / dtSec;
    const double pitchInst = std::atan(dyPixels / fy_) * kRadToDeg / dtSec;
    const double rollInst = daRadians * kRadToDeg / dtSec;

    const double alpha = 1.0 - std::exp(-dtSec / std::max(0.05, cfg_.tauSec));
    yawFiltered_ += alpha * (yawInst - yawFiltered_);
    pitchFiltered_ += alpha * (pitchInst - pitchFiltered_);
    rollFiltered_ += alpha * (rollInst - rollFiltered_);

    if (trackOk) {
        pushWindow(winYaw_, yawFiltered_);
        pushWindow(winPitch_, pitchFiltered_);
        pushWindow(winRoll_, rollFiltered_);
    } else {
        // Плохой кадр (мало точек/низкое качество) не должен загрязнять окно.
        winYaw_.clear();
        winPitch_.clear();
        winRoll_.clear();
        pushWindow(winYaw_, yawFiltered_);
        pushWindow(winPitch_, pitchFiltered_);
        pushWindow(winRoll_, rollFiltered_);
    }

    const double yaw = median(winYaw_);
    const double pitch = median(winPitch_);
    const double roll = median(winRoll_);
    const double maxAbs = std::max({std::abs(yaw), std::abs(pitch), std::abs(roll)});

    telem_.quality = 0.8 * telem_.quality + 0.2 * quality;
    const bool canLearn = trackOk && maxAbs <= cfg_.maxRateDps;
    telem_.valid = canLearn;

    if (canLearn) {
        timeSinceValid_ = 0.0;
        auto integrate = [&](double rate, double& cmd) {
            if (std::abs(rate) < cfg_.deadbandDps) return;
            cmd -= cfg_.integralGain * rate * dtSec;
        };
        integrate(yaw, telem_.yawCmdDps);
        integrate(pitch, telem_.pitchCmdDps);
        integrate(roll, telem_.rollCmdDps);
    } else {
        timeSinceValid_ += dtSec;
        // Если трекинг потерян надолго — плавно снимаем поправку.
        if (timeSinceValid_ > cfg_.holdSec) {
            const double decay = std::exp(-dtSec / 2.0);
            telem_.yawCmdDps *= decay;
            telem_.pitchCmdDps *= decay;
            telem_.rollCmdDps *= decay;
        }
    }

    // Утечка не даёт накапливать ошибку бесконечно.
    const double leak = std::exp(-dtSec / std::max(1.0, cfg_.leakTauSec));
    telem_.yawCmdDps = clampd(telem_.yawCmdDps * leak, -cfg_.maxDps, cfg_.maxDps);
    telem_.pitchCmdDps = clampd(telem_.pitchCmdDps * leak, -cfg_.maxDps, cfg_.maxDps);
    telem_.rollCmdDps = clampd(telem_.rollCmdDps * leak, -cfg_.maxDps, cfg_.maxDps);

    telem_.yawRateDps = yaw;
    telem_.pitchRateDps = pitch;
    telem_.rollRateDps = roll;
}

int16_t GyroCorrection::toInt16(double v) {
    v = std::max(-32768.0, std::min(32767.0, v));
    return static_cast<int16_t>(std::lround(v));
}

void GyroCorrection::assign(sbgc::GyroCorrection& c, int axis, int16_t value) {
    if (axis == 0) c.x = value;
    else if (axis == 1) c.y = value;
    else if (axis == 2) c.z = value;
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
