#pragma once

#include <opencv2/core.hpp>

// ========================= NEON WARP AFFINE =========================
// Собственная реализация warpAffine (bilinear, BORDER_CONSTANT) для 8UC1/8UC3.
// Возвращает false, если случай не поддержан — тогда нужно вызывать cv::warpAffine.
//
// Поддержано:
//   * src.depth() == CV_8U, channels == 1 или 3;
//   * интерполяция INTER_LINEAR (INTER_AREA трактуется как линейная);
//   * borderMode == BORDER_CONSTANT;
//   * flags с/без WARP_INVERSE_MAP.
//
// threads > 1 — распараллелить по строкам через пул OpenCV.
bool warpAffineNEON(const cv::Mat& src, cv::Mat& dst, const cv::Mat& M,
                    cv::Size dsize, int flags, int borderMode,
                    const cv::Scalar& borderValue, int threads = 1);

// true, если бинарник собран с NEON (AArch64/ARMv7+NEON).
bool warpAffineNEONAvailable();
