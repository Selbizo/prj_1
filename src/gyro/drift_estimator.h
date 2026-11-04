#pragma once

// Оценка углового дрейфа подвеса по изображению.
//
// Вход: межкадровое смещение dx/dy (пиксели полного кадра) и поворот da (рад),
// которые уже вычисляет VideoStabilizer. Выход: скорости рыскания/тангажа/крена
// (град/с), накопленный угол, достоверность и готовая поправка #75.
//
// Важно: это измерение движения камеры относительно сцены, а не абсолютный
// гироскопический дрейф. Медленное намеренное движение тоже попадёт в оценку.
// Поэтому дрейф выделяется фильтром нижних частот (tau), а знаки/оси и масштаб
// команды калибруются экспериментально.

#include "gyro/sbgc_protocol.h"

#include <string>

struct DriftConfig {
    double fovHdeg = 60.0;        // горизонтальный угол обзора, градусы
    double fovVdeg = 0.0;         // 0 — вычислить из fovH и соотношения сторон
    double tauSec = 2.0;          // постоянная времени ФНЧ для скорости дрейфа
    double deadbandDps = 0.02;    // мёртвая зона, град/с
    double maxDps = 2.0;          // ограничение поправки, град/с
    double minConfidence = 0.20;  // ниже — поправка не выдаётся
    double scaleUnitsPerDps = 16400.0; // 0.001 LSB при 16.4 LSB/(град/с)

    // Соответствие измеренных углов осям X/Y/Z команды #75.
    // По умолчанию: X — тангаж, Y — крен, Z — рыскание.
    int pitchAxis = 0;
    int yawAxis = 2;
    int rollAxis = 1;
    int pitchSign = 1;
    int yawSign = 1;
    int rollSign = 1;
};

struct DriftState {
    double pitchRateDps = 0.0;
    double yawRateDps = 0.0;
    double rollRateDps = 0.0;
    double pitchAngleDeg = 0.0;
    double yawAngleDeg = 0.0;
    double rollAngleDeg = 0.0;
    double confidence = 0.0;
    bool valid = false;
};

class DriftEstimator {
public:
    void configure(const DriftConfig& cfg, int frameWidth, int frameHeight);

    // dtSec — время между обработанными кадрами; quality — 0..1 (точки, inliers).
    DriftState update(double dxPixels, double dyPixels, double daRadians,
                      double dtSec, int points, double quality);

    const DriftState& state() const { return state_; }
    const DriftConfig& config() const { return cfg_; }
    bool configured() const { return configured_; }

    // Поправка для CMD_GYRO_CORRECTION: компенсирует измеренную скорость дрейфа.
    sbgc::GyroCorrection toGyroCorrection(const DriftState& s) const;

    static const char* axisName(int axis);

private:
    static int16_t clampInt16(double v);
    void assignAxis(sbgc::GyroCorrection& c, int axis, double units) const;

    DriftConfig cfg_;
    DriftState state_;
    bool configured_ = false;
    double fx_ = 1.0;
    double fy_ = 1.0;
    int width_ = 0;
    int height_ = 0;
};
