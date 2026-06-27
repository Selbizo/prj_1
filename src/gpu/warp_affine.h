#pragma once
#include "config/config.h"
#include <opencv2/core/ocl.hpp>

// ========================= WARP AFFINE ОПТИМИЗИРОВАННЫЙ =========================
void warpAffineOptimized(InputArray src, OutputArray dst, InputArray M, Size dsize,
                         int flags = INTER_LINEAR, int borderMode = BORDER_CONSTANT,
                         const Scalar& borderValue = Scalar(), bool useFP16 = false);
