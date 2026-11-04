#pragma once

// Минимальная оценка поправки #75 по оптическому потоку.
//
// dx/dy — пиксели полного кадра, da — радианы, dt — секунды.
// Пиксели -> угловая скорость -> ФНЧ -> интегратор -> единицы #75.

#include "gyro/sbgc_protocol.h"

struct GyroConfig {
    double fovHdeg = 60.0;             // горизонтальный FOV объектива
    double tauSec = 1.0;               // ФНЧ скорости
    double integralGain = 0.3;         // 1/с
    double deadbandDps = 0.02;         // мёртвая зона
    double maxRateDps = 3.0;           // выше — считаем движением, не дрейфом
    double maxDps = 2.0;               // лимит поправки
    double scaleUnitsPerDps = 16400.0; // единиц #75 на 1 град/с
    int pitchAxis = 0;                 // #75 X
    int yawAxis = 2;                   // #75 Z
    int rollAxis = 1;                  // #75 Y
    int pitchSign = 1;
    int yawSign = 1;
    int rollSign = 1;
};

struct GyroTelemetry {
    double yawRateDps = 0.0;
    double pitchRateDps = 0.0;
    double rollRateDps = 0.0;
    double yawCmdDps = 0.0;
    double pitchCmdDps = 0.0;
    double rollCmdDps = 0.0;
};

class GyroCorrection {
public:
    void configure(const GyroConfig& cfg, int frameWidth, int frameHeight);
    void update(double dxPixels, double dyPixels, double daRadians,
                double dtSec, int points);
    sbgc::GyroCorrection packet() const;
    const GyroTelemetry& telemetry() const { return telem_; }
    bool configured() const { return configured_; }

private:
    GyroConfig cfg_;
    GyroTelemetry telem_;
    bool configured_ = false;
    double fx_ = 1.0;
    double fy_ = 1.0;
};
