#pragma once

#ifndef CRUCIBLE_ULP_ANALYZER_CUH
#define CRUCIBLE_ULP_ANALYZER_CUH

// Crucible ULP Divergence Analyzer
//
// BitwiseCompareKernel reports *whether* outputs differ, but for
// publishable results we need *how much* they differ in floating-
// point terms.  The standard metric is ULP (Units in Last Place):
//
//   ulp_distance(a, b) = |reinterpret_as_int(a) - reinterpret_as_int(b)|
//
// ULP distance captures the number of representable floating-point
// values between a and b.  A 1-ULP difference is the smallest possible
// non-zero error; 2^23 ULP ≈ 1.0 relative error for FP32.
//
// This module provides:
//   1. Element-wise ULP distance kernel
//   2. ULP histogram kernel (distribution of error magnitudes)
//   3. Per-warp error statistics (for heatmap visualisation)
//
// The ULP histogram is directly usable as a paper figure:
//   "Distribution of cross-architecture numerical divergence"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <climits>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  ULP distance computation
//
//  For IEEE 754 floats, the integer reinterpretation has the
//  property that adjacent floats have adjacent integer values
//  (within the same sign).  ULP distance handles sign crossing
//  by converting to a "biased" representation first.
// ─────────────────────────────────────────────────────────────────

__device__ __forceinline__
int64_t float_to_biased_int(float x)
{
    // Map the IEEE-754 bit pattern to a monotonic unsigned key so that the
    // entire float line (−∞ … −0, +0 … +∞) becomes a single increasing total
    // order and |key(a) − key(b)| equals the number of representable floats
    // between a and b.
    //
    //   positives (sign bit 0): flip the sign bit       -> rank above all negatives
    //   negatives (sign bit 1): flip every bit (~)      -> reverse their descending order
    //
    // i.e. key = bits ^ ((bits >> 31) ? 0xFFFFFFFF : 0x80000000).
    //
    // The previous `0x80000000 - ix` form was broken: 0x80000000 is INT_MIN as
    // a signed literal, the subtraction overflowed, and the negative half kept
    // its descending order (−2.0 ranked above −1.0). Sign-crossing distances
    // and −0.0 vs +0.0 were therefore meaningless. This formulation cannot
    // overflow (all math in uint32) and is the standard radix-sort float key.
    const uint32_t bits = static_cast<uint32_t>(__float_as_int(x));
    const uint32_t mask = (bits >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return static_cast<int64_t>(bits ^ mask);
}

__device__ __forceinline__
uint64_t ulp_distance(float a, float b)
{
    // Handle NaN: infinite ULP distance
    if (isnan(a) || isnan(b))
        return UINT64_MAX;

    const int64_t ia = float_to_biased_int(a);
    const int64_t ib = float_to_biased_int(b);
    const int64_t diff = ia - ib;
    return static_cast<uint64_t>(diff < 0 ? -diff : diff);
}

// ─────────────────────────────────────────────────────────────────
//  Element-wise ULP distance kernel
//
//  Computes ulp_distance(a[i], b[i]) for all i and writes to
//  d_ulp_out.  Also computes per-warp statistics.
// ─────────────────────────────────────────────────────────────────

struct ULPStats {
    uint64_t max_ulp;
    uint64_t sum_ulp;          // for computing mean
    uint32_t count_nonzero;    // elements with ulp > 0
    uint32_t count_nan;        // elements where one is NaN and other isn't
    uint32_t total_elements;
};

template <int BLOCK_THREADS = 256>
__global__ void ULPDistanceKernel(
    const float*  __restrict__ d_a,
    const float*  __restrict__ d_b,
    uint64_t*     __restrict__ d_ulp_out,      // [num_elements] — per-element ULP
    ULPStats*     __restrict__ d_stats,         // [1] — aggregate stats
    uint32_t                   num_elements)
{
    uint64_t local_max = 0;
    uint64_t local_sum = 0;
    uint32_t local_nonzero = 0;
    uint32_t local_nan = 0;

    for (uint32_t i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         i < num_elements;
         i += gridDim.x * BLOCK_THREADS)
    {
        const uint64_t ulp = ulp_distance(d_a[i], d_b[i]);
        d_ulp_out[i] = ulp;

        if (ulp == UINT64_MAX) { ++local_nan; }
        else {
            local_max = (ulp > local_max) ? ulp : local_max;
            local_sum += ulp;
            if (ulp > 0) ++local_nonzero;
        }
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        uint64_t other_max = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
        local_max = (other_max > local_max) ? other_max : local_max;
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_nonzero += __shfl_down_sync(0xFFFFFFFF, local_nonzero, offset);
        local_nan += __shfl_down_sync(0xFFFFFFFF, local_nan, offset);
    }

    if ((threadIdx.x & 31) == 0)
    {
        // atomicMax for uint64 — use CAS loop
        unsigned long long* p_max =
            reinterpret_cast<unsigned long long*>(&d_stats->max_ulp);
        unsigned long long old_val = *p_max;
        while (local_max > old_val)
        {
            unsigned long long assumed = old_val;
            old_val = atomicCAS(p_max, assumed,
                                static_cast<unsigned long long>(local_max));
            if (old_val == assumed) break;
        }

        atomicAdd(reinterpret_cast<unsigned long long*>(&d_stats->sum_ulp),
                  static_cast<unsigned long long>(local_sum));
        atomicAdd(&d_stats->count_nonzero, local_nonzero);
        atomicAdd(&d_stats->count_nan,     local_nan);
    }
}

// ─────────────────────────────────────────────────────────────────
//  ULP histogram kernel
//
//  Buckets ULP distances into log2 bins:
//    bin 0:  ulp == 0          (exact match)
//    bin 1:  ulp ∈ [1, 1]      (1 ULP)
//    bin 2:  ulp ∈ [2, 3]      (2 ULP)
//    bin 3:  ulp ∈ [4, 7]      (3 ULP)
//    ...
//    bin k:  ulp ∈ [2^(k-1), 2^k - 1]
//    bin 33: ulp ≥ 2^32        (catastrophic)
//
//  This histogram is directly plottable as a publication figure:
//  x-axis = log2(ULP), y-axis = count.
// ─────────────────────────────────────────────────────────────────

static constexpr int ULP_HIST_BINS = 34;  // 0..33

__device__ __forceinline__
int ulp_to_bin(uint64_t ulp)
{
    if (ulp == 0) return 0;
    if (ulp == UINT64_MAX) return ULP_HIST_BINS - 1;  // NaN

    // floor(log2(ulp)) + 1, clamped to [1, 33]
    int bin = 0;
    uint64_t tmp = ulp;
    while (tmp >>= 1) ++bin;
    return (bin + 1 < ULP_HIST_BINS) ? bin + 1 : ULP_HIST_BINS - 1;
}

template <int BLOCK_THREADS = 256>
__global__ void ULPHistogramKernel(
    const uint64_t* __restrict__ d_ulp,          // [num_elements]
    uint32_t*       __restrict__ d_histogram,     // [ULP_HIST_BINS]
    uint32_t                     num_elements)
{
    // Shared memory histogram for block-level accumulation
    __shared__ uint32_t s_hist[ULP_HIST_BINS];

    if (threadIdx.x < ULP_HIST_BINS)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    for (uint32_t i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         i < num_elements;
         i += gridDim.x * BLOCK_THREADS)
    {
        const int bin = ulp_to_bin(d_ulp[i]);
        atomicAdd(s_hist + bin, 1u);
    }

    __syncthreads();

    // Flush block histogram to global
    if (threadIdx.x < ULP_HIST_BINS)
    {
        atomicAdd(d_histogram + threadIdx.x, s_hist[threadIdx.x]);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Error heatmap kernel
//
//  Computes per-tile error statistics for a 2D tensor (rows × dim).
//  Useful for visualising which embedding dimensions and which row
//  ranges exhibit the most cross-architecture divergence.
//
//  Output: a (num_row_tiles × num_dim_tiles) matrix of mean ULP.
// ─────────────────────────────────────────────────────────────────

template <int TILE_ROWS = 64, int TILE_COLS = 16>
__global__ void ErrorHeatmapKernel(
    const float*  __restrict__ d_a,           // [rows × cols]
    const float*  __restrict__ d_b,
    float*        __restrict__ d_heatmap,     // [row_tiles × col_tiles]
    uint32_t                   rows,
    uint32_t                   cols,
    uint32_t                   row_tiles,
    uint32_t                   col_tiles)
{
    const uint32_t tile_r = blockIdx.y;
    const uint32_t tile_c = blockIdx.x;

    if (tile_r >= row_tiles || tile_c >= col_tiles) return;

    const uint32_t r_start = tile_r * TILE_ROWS;
    const uint32_t c_start = tile_c * TILE_COLS;
    const uint32_t r_end   = min(r_start + TILE_ROWS, rows);
    const uint32_t c_end   = min(c_start + TILE_COLS, cols);

    // Each thread accumulates over a portion of the tile
    uint64_t local_sum = 0;
    uint32_t local_count = 0;

    for (uint32_t r = r_start + threadIdx.y; r < r_end; r += blockDim.y)
    {
        for (uint32_t c = c_start + threadIdx.x; c < c_end; c += blockDim.x)
        {
            const uint32_t idx = r * cols + c;
            const uint64_t ulp = ulp_distance(d_a[idx], d_b[idx]);
            if (ulp != UINT64_MAX)
            {
                local_sum += ulp;
                ++local_count;
            }
        }
    }

    // Reduce within block
    __shared__ uint64_t s_sum;
    __shared__ uint32_t s_count;
    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        s_sum = 0;
        s_count = 0;
    }
    __syncthreads();

    atomicAdd(reinterpret_cast<unsigned long long*>(&s_sum),
              static_cast<unsigned long long>(local_sum));
    atomicAdd(&s_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0)
    {
        const float mean_ulp = (s_count > 0)
            ? static_cast<float>(s_sum) / static_cast<float>(s_count)
            : 0.0f;
        d_heatmap[tile_r * col_tiles + tile_c] = mean_ulp;
    }
}

}  // namespace Crucible

#endif  // CRUCIBLE_ULP_ANALYZER_CUH
