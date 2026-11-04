#include "gpu/warp_affine_neon.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(__aarch64__)
#define WARP_NEON_ENABLED 1
#include <arm_neon.h>
#else
#define WARP_NEON_ENABLED 0
#endif

using namespace cv;

namespace {

// Координаты источника — та же fixed-point сетка, что в cv::warpAffine:
// Q10.10 (AB_BITS=10), затем дробная часть 1/32 пикселя (INTER_BITS=5).
// round_delta = AB_SCALE / INTER_TAB_SIZE / 2 = 16 — округление до 1/32.
constexpr int kAbBits = 10;
constexpr int kInterBits = 5;
constexpr int kInterScale = 1 << kInterBits;              // 32
constexpr int kRoundDelta = (1 << kAbBits) / kInterScale / 2;  // 16

inline uint64_t loadU64(const uchar* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Ровно 3 байта, без «залезания» в следующий пиксель (перекрывающиеся
// 4-байтовые записи в параллельном варпе недопустимы).
inline void storeBGR(uchar* p, uint32_t v) {
    const uint16_t lo = (uint16_t)v;
    std::memcpy(p, &lo, sizeof(lo));
    p[2] = (uchar)(v >> 16);
}

struct Border8 {
    uchar v[4];
};

// Прямой доступ к пикселям без Mat::ptr/at (в отладочной сборке они делают
// проверки границ на каждый пиксель, что для внутреннего цикла недопустимо).
struct SrcView {
    const uchar* data;
    size_t step;
    int w, h;
};

Border8 makeBorder8(const Scalar& borderValue, int cn) {
    Border8 b{{0, 0, 0, 0}};
    for (int c = 0; c < cn; ++c) {
        b.v[c] = saturate_cast<uchar>(borderValue[c]);
    }
    return b;
}

// Скалярный bilinear для одного пикселя (границы кадра и «хвост» строки).
inline void samplePixel8UC1(const SrcView& src, int ix, int fx, int iy, int fy,
                            const Border8& border, uchar* out) {
    const int w = src.w, h = src.h;
    const int wx[2] = {kInterScale - fx, fx};
    const int wy[2] = {kInterScale - fy, fy};
    int sum = 0;
    for (int dy = 0; dy < 2; ++dy) {
        const int py = iy + dy;
        if ((unsigned)py >= (unsigned)h) {
            sum += wy[dy] * kInterScale * border.v[0];
            continue;
        }
        const uchar* row = src.data + (size_t)py * src.step;
        for (int dx = 0; dx < 2; ++dx) {
            const int weight = wx[dx] * wy[dy];
            if (!weight) continue;
            const int px = ix + dx;
            const uchar v = ((unsigned)px < (unsigned)w) ? row[px] : border.v[0];
            sum += weight * v;
        }
    }
    *out = (uchar)((sum + 512) >> 10);
}

inline void samplePixel8UC3(const SrcView& src, int ix, int fx, int iy, int fy,
                            const Border8& border, uchar* out) {
    const int w = src.w, h = src.h;
    const int wx[2] = {kInterScale - fx, fx};
    const int wy[2] = {kInterScale - fy, fy};
    int s0 = 0, s1 = 0, s2 = 0;
    for (int dy = 0; dy < 2; ++dy) {
        const int py = iy + dy;
        if ((unsigned)py >= (unsigned)h) {
            const int wr = wy[dy] * kInterScale;
            s0 += wr * border.v[0];
            s1 += wr * border.v[1];
            s2 += wr * border.v[2];
            continue;
        }
        const uchar* row = src.data + (size_t)py * src.step;
        for (int dx = 0; dx < 2; ++dx) {
            const int weight = wx[dx] * wy[dy];
            if (!weight) continue;
            const int px = ix + dx;
            if ((unsigned)px < (unsigned)w) {
                const uchar* p = row + 3 * px;
                s0 += weight * p[0];
                s1 += weight * p[1];
                s2 += weight * p[2];
            } else {
                s0 += weight * border.v[0];
                s1 += weight * border.v[1];
                s2 += weight * border.v[2];
            }
        }
    }
    out[0] = (uchar)((s0 + 512) >> 10);
    out[1] = (uchar)((s1 + 512) >> 10);
    out[2] = (uchar)((s2 + 512) >> 10);
}

#if WARP_NEON_ENABLED

inline int32_t hmin_s32(int32x4_t v) {
#if defined(__aarch64__)
    return vminvq_s32(v);
#else
    int32x2_t t = vpmin_s32(vget_low_s32(v), vget_high_s32(v));
    t = vpmin_s32(t, t);
    return vget_lane_s32(t, 0);
#endif
}

inline int32_t hmax_s32(int32x4_t v) {
#if defined(__aarch64__)
    return vmaxvq_s32(v);
#else
    int32x2_t t = vpmax_s32(vget_low_s32(v), vget_high_s32(v));
    t = vpmax_s32(t, t);
    return vget_lane_s32(t, 0);
#endif
}

// ---------------------------------------------------------------- 8UC1 ----

void warpRows8UC1(const SrcView& src, uchar* ddata, size_t dstep, int dw,
                  const double m[6], const int* adelta, const int* bdelta,
                  const Border8& border, int yBegin, int yEnd) {
    const int sw = src.w, sh = src.h;
    const int32x4_t v31 = vdupq_n_s32(kInterScale - 1);

    for (int y = yBegin; y < yEnd; ++y) {
        const int X0 = (int)std::llround((m[1] * y + m[2]) * (1 << kAbBits)) + kRoundDelta;
        const int Y0 = (int)std::llround((m[4] * y + m[5]) * (1 << kAbBits)) + kRoundDelta;
        uchar* drow = ddata + (size_t)y * dstep;
        int x = 0;

        for (; x + 4 <= dw; x += 4) {
            const int32x4_t X10 = vaddq_s32(vld1q_s32(adelta + x), vdupq_n_s32(X0));
            const int32x4_t Y10 = vaddq_s32(vld1q_s32(bdelta + x), vdupq_n_s32(Y0));
            const int32x4_t ix = vshrq_n_s32(X10, kAbBits);
            const int32x4_t iy = vshrq_n_s32(Y10, kAbBits);
            const int32x4_t fx = vandq_s32(vshrq_n_s32(X10, kAbBits - kInterBits), v31);
            const int32x4_t fy = vandq_s32(vshrq_n_s32(Y10, kAbBits - kInterBits), v31);

            const int ixmin = hmin_s32(ix), ixmax = hmax_s32(ix);
            const int iymin = hmin_s32(iy), iymax = hmax_s32(iy);
            const bool interior = ixmin >= 0 && ixmax <= sw - 2 &&
                                  iymin >= 0 && iymax <= sh - 2;

            if (interior) {
                int ixa[4], iya[4];
                vst1q_s32(ixa, ix);
                vst1q_s32(iya, iy);
                uint32_t p00[4], p01[4], p10[4], p11[4];
                for (int k = 0; k < 4; ++k) {
                    const uchar* r0 = src.data + (size_t)iya[k] * src.step + ixa[k];
                    const uchar* r1 = r0 + src.step;
                    uint16_t v0, v1;
                    std::memcpy(&v0, r0, sizeof(v0));
                    std::memcpy(&v1, r1, sizeof(v1));
                    p00[k] = v0 & 0xFF;
                    p01[k] = v0 >> 8;
                    p10[k] = v1 & 0xFF;
                    p11[k] = v1 >> 8;
                }
                const uint32x4_t V00 = vld1q_u32(p00);
                const uint32x4_t V01 = vld1q_u32(p01);
                const uint32x4_t V10 = vld1q_u32(p10);
                const uint32x4_t V11 = vld1q_u32(p11);
                const uint32x4_t wx0 = vreinterpretq_u32_s32(
                    vsubq_s32(vdupq_n_s32(kInterScale), fx));
                const uint32x4_t wx1 = vreinterpretq_u32_s32(fx);
                const uint32x4_t wy0 = vreinterpretq_u32_s32(
                    vsubq_s32(vdupq_n_s32(kInterScale), fy));
                const uint32x4_t wy1 = vreinterpretq_u32_s32(fy);

                const uint32x4_t h0 = vaddq_u32(vmulq_u32(V00, wx0), vmulq_u32(V01, wx1));
                const uint32x4_t h1 = vaddq_u32(vmulq_u32(V10, wx0), vmulq_u32(V11, wx1));
                uint32x4_t res = vaddq_u32(vmulq_u32(h0, wy0), vmulq_u32(h1, wy1));
                res = vshrq_n_u32(vaddq_u32(res, vdupq_n_u32(512)), 10);

                drow[x]     = (uchar)vgetq_lane_u32(res, 0);
                drow[x + 1] = (uchar)vgetq_lane_u32(res, 1);
                drow[x + 2] = (uchar)vgetq_lane_u32(res, 2);
                drow[x + 3] = (uchar)vgetq_lane_u32(res, 3);
            } else {
                int ixa[4], iya[4], fxa[4], fya[4];
                vst1q_s32(ixa, ix);
                vst1q_s32(iya, iy);
                vst1q_s32(fxa, fx);
                vst1q_s32(fya, fy);
                for (int k = 0; k < 4; ++k) {
                    samplePixel8UC1(src, ixa[k], fxa[k], iya[k], fya[k],
                                    border, drow + x + k);
                }
            }
        }

        for (; x < dw; ++x) {
            const int X10 = X0 + adelta[x];
            const int Y10 = Y0 + bdelta[x];
            const int ix = X10 >> kAbBits;
            const int iy = Y10 >> kAbBits;
            const int fx = (X10 >> (kAbBits - kInterBits)) & (kInterScale - 1);
            const int fy = (Y10 >> (kAbBits - kInterBits)) & (kInterScale - 1);
            samplePixel8UC1(src, ix, fx, iy, fy, border, drow + x);
        }
    }
}

// ---------------------------------------------------------------- 8UC3 ----

// Горизонтальная интерполяция двух строк + вертикальная. 1/32 px, >> 10.
inline uint32x4_t bilinear3(uint32x4_t p00, uint32x4_t p01,
                            uint32x4_t p10, uint32x4_t p11,
                            uint32x4_t w0, uint32x4_t w1,
                            uint32x4_t u0, uint32x4_t u1) {
    const uint32x4_t h0 = vaddq_u32(vmulq_u32(p00, w0), vmulq_u32(p01, w1));
    const uint32x4_t h1 = vaddq_u32(vmulq_u32(p10, w0), vmulq_u32(p11, w1));
    uint32x4_t v = vaddq_u32(vmulq_u32(h0, u0), vmulq_u32(h1, u1));
    return vshrq_n_u32(vaddq_u32(v, vdupq_n_u32(512)), 10);
}

void warpRows8UC3(const SrcView& src, uchar* ddata, size_t dstep, int dw,
                  const double m[6], const int* adelta, const int* bdelta,
                  const Border8& border, int yBegin, int yEnd) {
    const int sw = src.w, sh = src.h;
    const uint32x4_t vmask = vdupq_n_u32(0xFF);
    const int32x4_t v31 = vdupq_n_s32(kInterScale - 1);

    for (int y = yBegin; y < yEnd; ++y) {
        const int X0 = (int)std::llround((m[1] * y + m[2]) * (1 << kAbBits)) + kRoundDelta;
        const int Y0 = (int)std::llround((m[4] * y + m[5]) * (1 << kAbBits)) + kRoundDelta;
        uchar* drow = ddata + (size_t)y * dstep;
        int x = 0;

        // x + 4 < dw: последняя 4-пиксельная группа пишет 4 байта на пиксель
        // (перекрытие со следующим пикселем), поэтому её и хвост считаем скалярно.
        for (; x + 4 < dw; x += 4) {
            const int32x4_t X10 = vaddq_s32(vld1q_s32(adelta + x), vdupq_n_s32(X0));
            const int32x4_t Y10 = vaddq_s32(vld1q_s32(bdelta + x), vdupq_n_s32(Y0));
            const int32x4_t ix = vshrq_n_s32(X10, kAbBits);
            const int32x4_t iy = vshrq_n_s32(Y10, kAbBits);
            const int32x4_t fx = vandq_s32(vshrq_n_s32(X10, kAbBits - kInterBits), v31);
            const int32x4_t fy = vandq_s32(vshrq_n_s32(Y10, kAbBits - kInterBits), v31);

            const int ixmin = hmin_s32(ix), ixmax = hmax_s32(ix);
            const int iymin = hmin_s32(iy), iymax = hmax_s32(iy);
            // ix <= sw-3 — чтобы 8-байтовое чтение двух пикселей не вышло за строку.
            const bool interior = ixmin >= 0 && ixmax <= sw - 3 &&
                                  iymin >= 0 && iymax <= sh - 2;

            int ixa[4], iya[4];
            vst1q_s32(ixa, ix);
            vst1q_s32(iya, iy);

            if (interior) {
                // Один 8-байтовый load на строку даёт сразу два соседних
                // пикселя (BGR+BGR): вдвое меньше обращений к памяти, чем
                // четыре отдельных 4-байтовых load на пиксель.
                uint64_t q00[4], q10[4];
                for (int k = 0; k < 4; ++k) {
                    const uchar* r0 = src.data + (size_t)iya[k] * src.step + 3 * ixa[k];
                    q00[k] = loadU64(r0);
                    q10[k] = loadU64(r0 + src.step);
                }
                const uint64x2_t q00a = vld1q_u64(q00);
                const uint64x2_t q00b = vld1q_u64(q00 + 2);
                const uint64x2_t q10a = vld1q_u64(q10);
                const uint64x2_t q10b = vld1q_u64(q10 + 2);
                const uint32x4_t V00 = vcombine_u32(vmovn_u64(q00a), vmovn_u64(q00b));
                const uint32x4_t V01 = vcombine_u32(vmovn_u64(vshrq_n_u64(q00a, 24)),
                                                    vmovn_u64(vshrq_n_u64(q00b, 24)));
                const uint32x4_t V10 = vcombine_u32(vmovn_u64(q10a), vmovn_u64(q10b));
                const uint32x4_t V11 = vcombine_u32(vmovn_u64(vshrq_n_u64(q10a, 24)),
                                                    vmovn_u64(vshrq_n_u64(q10b, 24)));

                const uint32x4_t wx0 = vreinterpretq_u32_s32(
                    vsubq_s32(vdupq_n_s32(kInterScale), fx));
                const uint32x4_t wx1 = vreinterpretq_u32_s32(fx);
                const uint32x4_t wy0 = vreinterpretq_u32_s32(
                    vsubq_s32(vdupq_n_s32(kInterScale), fy));
                const uint32x4_t wy1 = vreinterpretq_u32_s32(fy);

                const uint32x4_t b = bilinear3(vandq_u32(V00, vmask),
                                               vandq_u32(V01, vmask),
                                               vandq_u32(V10, vmask),
                                               vandq_u32(V11, vmask), wx0, wx1, wy0, wy1);
                const uint32x4_t g = bilinear3(
                    vandq_u32(vshrq_n_u32(V00, 8), vmask),
                    vandq_u32(vshrq_n_u32(V01, 8), vmask),
                    vandq_u32(vshrq_n_u32(V10, 8), vmask),
                    vandq_u32(vshrq_n_u32(V11, 8), vmask), wx0, wx1, wy0, wy1);
                const uint32x4_t r = bilinear3(
                    vandq_u32(vshrq_n_u32(V00, 16), vmask),
                    vandq_u32(vshrq_n_u32(V01, 16), vmask),
                    vandq_u32(vshrq_n_u32(V10, 16), vmask),
                    vandq_u32(vshrq_n_u32(V11, 16), vmask), wx0, wx1, wy0, wy1);

                const uint32x4_t pix = vorrq_u32(b, vorrq_u32(vshlq_n_u32(g, 8),
                                                              vshlq_n_u32(r, 16)));
                uchar* o = drow + 3 * x;
                storeBGR(o,     vgetq_lane_u32(pix, 0));
                storeBGR(o + 3, vgetq_lane_u32(pix, 1));
                storeBGR(o + 6, vgetq_lane_u32(pix, 2));
                storeBGR(o + 9, vgetq_lane_u32(pix, 3));
            } else {
                int fxa[4], fya[4];
                vst1q_s32(fxa, fx);
                vst1q_s32(fya, fy);
                for (int k = 0; k < 4; ++k) {
                    samplePixel8UC3(src, ixa[k], fxa[k], iya[k], fya[k],
                                    border, drow + 3 * (x + k));
                }
            }
        }

        for (; x < dw; ++x) {
            const int X10 = X0 + adelta[x];
            const int Y10 = Y0 + bdelta[x];
            const int ix = X10 >> kAbBits;
            const int iy = Y10 >> kAbBits;
            const int fx = (X10 >> (kAbBits - kInterBits)) & (kInterScale - 1);
            const int fy = (Y10 >> (kAbBits - kInterBits)) & (kInterScale - 1);
            samplePixel8UC3(src, ix, fx, iy, fy, border, drow + 3 * x);
        }
    }
}

#endif  // WARP_NEON_ENABLED

}  // namespace

bool warpAffineNEONAvailable() {
    return WARP_NEON_ENABLED != 0;
}

bool warpAffineNEON(const Mat& src, Mat& dst, const Mat& M,
                    Size dsize, int flags, int borderMode,
                    const Scalar& borderValue, int threads) {
#if !WARP_NEON_ENABLED
    (void)src; (void)dst; (void)M; (void)dsize; (void)flags;
    (void)borderMode; (void)borderValue; (void)threads;
    return false;
#else
    if (src.empty() || src.depth() != CV_8U) return false;
    const int cn = src.channels();
    if (cn != 1 && cn != 3) return false;

    const int interp = flags & INTER_MAX;
    if (interp != INTER_LINEAR && interp != INTER_AREA) return false;
    if (borderMode != BORDER_CONSTANT) return false;
    if (flags & WARP_RELATIVE_MAP) return false;
    if (M.rows != 2 || M.cols != 3) return false;

    if (dsize.width <= 0 || dsize.height <= 0) dsize = src.size();
    if (dsize.width < 1 || dsize.height < 1) return false;

    // Матрица dst -> src.
    Mat inv;
    if (flags & WARP_INVERSE_MAP) {
        inv = M;
    } else {
        invertAffineTransform(M, inv);
    }
    Mat invd;
    inv.convertTo(invd, CV_64F);
    if (!invd.isContinuous()) return false;
    const double* p = invd.ptr<double>();
    const double det = p[0] * p[4] - p[1] * p[3];
    if (!std::isfinite(det) || std::fabs(det) < 1e-12) return false;
    double m[6];
    for (int i = 0; i < 6; ++i) {
        m[i] = p[i];
        if (!std::isfinite(m[i])) return false;
    }

    if (dst.size() != dsize || dst.type() != src.type()) {
        dst.create(dsize, src.type());
    }
    Mat out;
    if (dst.isContinuous()) {
        out = dst;
    } else {
        out.create(dsize, src.type());
    }

    const Border8 border = makeBorder8(borderValue, cn);

    // adelta[x] = round(m[0]*x*1024), bdelta[x] = round(m[3]*x*1024) —
    // ровно как в WarpAffineInvoker из OpenCV. Считается один раз на кадр.
    std::vector<int> adelta(dsize.width), bdelta(dsize.width);
    for (int x = 0; x < dsize.width; ++x) {
        adelta[x] = (int)std::llround(m[0] * x * (1 << kAbBits));
        bdelta[x] = (int)std::llround(m[3] * x * (1 << kAbBits));
    }

    const SrcView view{src.data, src.step, src.cols, src.rows};
    const bool useThreads = threads > 1 && dsize.height >= 64;
    auto runRows = [&](int y0, int y1) {
        if (cn == 1)
            warpRows8UC1(view, out.data, out.step, out.cols, m,
                         adelta.data(), bdelta.data(), border, y0, y1);
        else
            warpRows8UC3(view, out.data, out.step, out.cols, m,
                         adelta.data(), bdelta.data(), border, y0, y1);
    };

    if (useThreads) {
        parallel_for_(Range(0, dsize.height), [&](const Range& range) {
            runRows(range.start, range.end);
        }, threads);
    } else {
        runRows(0, dsize.height);
    }

    if (out.data != dst.data) out.copyTo(dst);
    return true;
#endif
}
