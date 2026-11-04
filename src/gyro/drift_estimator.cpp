#include "gyro/drift_estimator.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kRadToDeg = 180.0 / 3.14159265358979323846;

double clampd(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}
} // namespace

void DriftEstimator::configure(const DriftConfig& cfg, int frameWidth, int frameHeight) {
    cfg_ = cfg;
    width_ = frameWidth;
    height_ = frameHeight;
    const double fovH = clampd(cfg_.fovHdeg, 5.0, 170.0) * 3.14159265358979323846 / 180.0;
    double fovV = cfg_.fovVdeg;
    if (fovV <= 0.0 && frameWidth > 0 && frameHeight > 0) {
        // Из горизонтального FOV и соотношения сторон: для квадратных пикселей
        // fx = fy, поэтому вертикальный FOV вычисляется через тангенс.
        double tanH = std::tan(fovH / 2.0);
        fovV = 2.0 * std::atan(tanH * static_cast<double>(frameHeight) / frameWidth);
    } else {
        fovV = clampd(fovV, 5.0, 170.0) * 3.14159265358979323846 / 180.0;
    }
    fx_ = (frameWidth > 0) ? (frameWidth / 2.0) / std::tan(fovH / 2.0) : 1.0;
    fy_ = (frameHeight > 0) ? (frameHeight / 2.0) / std::tan(fovV / 2.0) : 1.0;
    if (fx_ < 1.0) fx_ = 1.0;
    if (fy_ < 1.0) fy_ = 1.0;
    state_ = DriftState{};
    configured_ = true;
}

DriftState DriftEstimator::update(double dxPixels, double dyPixels, double daRadians,
                                  double dtSec, int points, double quality) {
    if (!configured_ || dtSec <= 1e-4 || dtSec > 0.5) {
        state_.valid = false;
        return state_;
    }

    quality = clampd(quality, 0.0, 1.0);
    // Мало точек — измерению нельзя доверять, но полностью обнулять фильтр
    // не нужно: иначе на коротких просадках поправка дёргается.
    if (points < 6) quality = 0.0;

    const double yawInst = std::atan(dxPixels / fx_) * kRadToDeg / dtSec;
    const double pitchInst = std::atan(dyPixels / fy_) * kRadToDeg / dtSec;
    const double rollInst = daRadians * kRadToDeg / dtSec;

    // ФНЧ первого порядка: alpha = 1 - exp(-dt/tau). При quality=0 фильтр
    // не обновляется (держит последнюю оценку), но valid сбрасывается.
    const double tau = std::max(0.05, cfg_.tauSec);
    const double alpha = (1.0 - std::exp(-dtSec / tau)) * quality;
    state_.yawRateDps += alpha * (yawInst - state_.yawRateDps);
    state_.pitchRateDps += alpha * (pitchInst - state_.pitchRateDps);
    state_.rollRateDps += alpha * (rollInst - state_.rollRateDps);

    state_.yawAngleDeg += state_.yawRateDps * dtSec;
    state_.pitchAngleDeg += state_.pitchRateDps * dtSec;
    state_.rollAngleDeg += state_.rollRateDps * dtSec;

    state_.confidence = 0.8 * state_.confidence + 0.2 * quality;
    state_.valid = (quality > 0.0);
    return state_;
}

int16_t DriftEstimator::clampInt16(double v) {
    v = std::max(-32768.0, std::min(32767.0, v));
    return static_cast<int16_t>(std::lround(v));
}

void DriftEstimator::assignAxis(sbgc::GyroCorrection& c, int axis, double units) const {
    const int16_t value = clampInt16(units);
    if (axis == 0) c.x = value;
    else if (axis == 1) c.y = value;
    else if (axis == 2) c.z = value;
}

sbgc::GyroCorrection DriftEstimator::toGyroCorrection(const DriftState& s) const {
    sbgc::GyroCorrection c;
    c.imuType = 0;
    if (!s.valid || s.confidence < cfg_.minConfidence) return c;

    auto correctionUnits = [&](double rateDps, double deadbandDps, double sign) -> double {
        if (std::abs(rateDps) < deadbandDps) return 0.0;
        double dps = clampd(rateDps, -cfg_.maxDps, cfg_.maxDps);
        // Знак минус: поправка должна компенсировать измеренный дрейф.
        return -dps * cfg_.scaleUnitsPerDps * sign;
    };

    assignAxis(c, cfg_.pitchAxis, correctionUnits(s.pitchRateDps, cfg_.deadbandDps, cfg_.pitchSign));
    assignAxis(c, cfg_.yawAxis, correctionUnits(s.yawRateDps, cfg_.deadbandDps, cfg_.yawSign));
    assignAxis(c, cfg_.rollAxis, correctionUnits(s.rollRateDps, cfg_.deadbandDps, cfg_.rollSign));
    c.heading = 0;
    return c;
}

const char* DriftEstimator::axisName(int axis) {
    switch (axis) {
        case 0: return "X";
        case 1: return "Y";
        case 2: return "Z";
        default: return "?";
    }
}
