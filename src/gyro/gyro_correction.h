#pragma once

// Внешний контур: оптический поток -> оценка теплового дрейфа -> пакет #75.
// Внутренний контур (стабилизация, гироскоп, моторы) не затрагивается.
//
// 120 Hz: update() вызывается на каждом кадре, packet() можно отправлять
// каждый кадр. Оценка bias намеренно медленная (тепловой дрейф), поэтому
// используется робастное скользящее окно + интегратор с утечкой.

#include "gyro/sbgc_protocol.h"

#include <deque>

struct GyroConfig {
    double fovHdeg = 60.0;             // горизонтальный FOV
    double tauSec = 0.5;               // ФНЧ перед окном
    int windowSamples = 60;            // окно медианы: 60 кадров ~ 0.5 с при 120 Hz
    double integralGain = 0.5;         // 1/с
    double leakTauSec = 60.0;          // утечка накопленной поправки, с
    double deadbandDps = 0.02;         // мёртвая зона
    double maxRateDps = 3.0;           // выше — движение, не дрейф
    double maxDps = 2.0;               // предел поправки
    double scaleUnitsPerDps = 16400.0; // единиц #75 на 1 град/с
    int pitchAxis = 0;                 // #75 X
    int yawAxis = 2;                   // #75 Z
    int rollAxis = 1;                  // #75 Y
    int pitchSign = 1;
    int yawSign = 1;
    int rollSign = 1;
    int minPoints = 8;                 // меньше точек — не учимся
    double minQuality = 0.20;          // ниже — не учимся
    double holdSec = 2.0;              // сколько держать поправку без трекинга
};

struct GyroTelemetry {
    double yawRateDps = 0.0;
    double pitchRateDps = 0.0;
    double rollRateDps = 0.0;
    double yawCmdDps = 0.0;
    double pitchCmdDps = 0.0;
    double rollCmdDps = 0.0;
    double quality = 0.0;
    bool valid = false;
};

class GyroCorrection {
public:
    void configure(const GyroConfig& cfg, int frameWidth, int frameHeight);
    // dx/dy — пиксели полного кадра, da — радианы, dt — секунды.
    // quality — 0..1 от трекера (точки + inliers).
    void update(double dxPixels, double dyPixels, double daRadians,
                double dtSec, int points, double quality);
    sbgc::GyroCorrection packet() const;
    const GyroTelemetry& telemetry() const { return telem_; }
    bool configured() const { return configured_; }

private:
    void pushWindow(std::deque<double>& window, double value) const;
    static double median(std::deque<double> window);
    static double clampd(double v, double lo, double hi);
    static int16_t toInt16(double v);
    static void assign(sbgc::GyroCorrection& c, int axis, int16_t value);

    GyroConfig cfg_;
    GyroTelemetry telem_;
    std::deque<double> winYaw_;
    std::deque<double> winPitch_;
    std::deque<double> winRoll_;
    double yawFiltered_ = 0.0;
    double pitchFiltered_ = 0.0;
    double rollFiltered_ = 0.0;
    double timeSinceValid_ = 0.0;
    bool configured_ = false;
    double fx_ = 1.0;
    double fy_ = 1.0;
};
