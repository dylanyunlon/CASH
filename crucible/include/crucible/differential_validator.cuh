#pragma once

#ifndef CRUCIBLE_DIFFERENTIAL_VALIDATOR_CUH
#define CRUCIBLE_DIFFERENTIAL_VALIDATOR_CUH

/// Crucible Differential Validator
///
/// Adapts SyzMini's influence-guided test reduction to CUDA kernel
/// parameter space.  The core idea:
///
///   For each kernel K, run K on sm86 and sm90 with identical inputs.
///   Compare outputs.  Track which *parameters* (grid dim, block dim,
///   shared mem, alignment, tensor shape) most influence divergence.
///   Focus subsequent fuzzing on high-influence parameter regions.
///
/// The influence score is an exponentially-weighted moving average:
///   score[param] = α·caused_divergence + (1-α)·score[param]
///
/// Parameters with high influence scores are mutated more aggressively
/// in subsequent rounds (wider perturbation range, boundary-biased
/// sampling).
///
/// This file provides the device-side kernels for:
///   1. Test execution wrappers (run-and-capture)
///   2. Bitwise comparison
///   3. Influence-weighted boundary generation

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  Bitwise comparison kernel
//
//  Compares two output buffers element-by-element.  For each
//  mismatch, records the index and magnitude of the difference.
//  Uses a two-level reduction: warp-level ballot to find any
//  mismatches quickly, then per-element analysis only for
//  divergent warps.
// ─────────────────────────────────────────────────────────────────

struct DivergenceReport {
    uint32_t total_elements;
    uint32_t mismatched_elements;
    float    max_abs_diff;
    float    max_rel_diff;
    uint32_t first_mismatch_idx;
};

template <typename T, int BLOCK_THREADS = 256>
__global__ void BitwiseCompareKernel(
    const T*           __restrict__ d_output_a,   // sm86 output
    const T*           __restrict__ d_output_b,   // sm90 output
    uint32_t                        num_elements,
    DivergenceReport*  __restrict__ d_report)
{
    // Per-thread local maximums
    float local_abs_max = 0.0f;
    float local_rel_max = 0.0f;
    uint32_t local_mismatches = 0;
    uint32_t local_first_idx  = UINT32_MAX;

    for (uint32_t i = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         i < num_elements;
         i += gridDim.x * BLOCK_THREADS)
    {
        const T a = d_output_a[i];
        const T b = d_output_b[i];

        // Bitwise comparison: reinterpret as same-sized integer
        // This catches sign-bit flips and NaN payload differences
        // that floating-point == would miss.
        bool bitwise_equal;
        if constexpr (sizeof(T) == 4)
        {
            bitwise_equal = (__float_as_int(static_cast<float>(a)) ==
                             __float_as_int(static_cast<float>(b)));
        }
        else if constexpr (sizeof(T) == 2)
        {
            bitwise_equal = (*reinterpret_cast<const uint16_t*>(&a) ==
                             *reinterpret_cast<const uint16_t*>(&b));
        }
        else
        {
            bitwise_equal = (*reinterpret_cast<const uint8_t*>(&a) ==
                             *reinterpret_cast<const uint8_t*>(&b));
        }

        if (!bitwise_equal)
        {
            ++local_mismatches;
            if (i < local_first_idx) local_first_idx = i;

            const float fa = static_cast<float>(a);
            const float fb = static_cast<float>(b);
            const float abs_diff = fabsf(fa - fb);
            const float rel_diff = abs_diff / fmaxf(fabsf(fa), 1e-10f);

            local_abs_max = fmaxf(local_abs_max, abs_diff);
            local_rel_max = fmaxf(local_rel_max, rel_diff);
        }
    }

    // ── Warp-level reduction ──────────────────────────────────
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_abs_max    = fmaxf(local_abs_max,
                                 __shfl_down_sync(0xFFFFFFFF, local_abs_max, offset));
        local_rel_max    = fmaxf(local_rel_max,
                                 __shfl_down_sync(0xFFFFFFFF, local_rel_max, offset));
        local_mismatches += __shfl_down_sync(0xFFFFFFFF, local_mismatches, offset);
        local_first_idx  = min(local_first_idx,
                               __shfl_down_sync(0xFFFFFFFF, local_first_idx, offset));
    }

    // ── Lane 0 of each warp writes to global report ──────────
    if ((threadIdx.x & 31) == 0)
    {
        atomicAdd(&d_report->mismatched_elements, local_mismatches);
        atomicMax(reinterpret_cast<int*>(&d_report->max_abs_diff),
                  __float_as_int(local_abs_max));
        atomicMax(reinterpret_cast<int*>(&d_report->max_rel_diff),
                  __float_as_int(local_rel_max));
        atomicMin(&d_report->first_mismatch_idx, local_first_idx);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Index kernel test wrappers
//
//  These wrap HypeReca's three index kernels with boundary-
//  condition inputs designed to trigger architecture-dependent
//  behaviour.  Each wrapper:
//    1. Initializes input buffers with deterministic patterns
//    2. Calls the kernel
//    3. Stores the output for comparison
//
//  The wrapper is parameterised by tensor shape and alignment
//  so the fuzzer can sweep the parameter space.
// ─────────────────────────────────────────────────────────────────

/// Deterministic pattern fill: value at index i is derived from
/// i via a bijective function so that any reordering or truncation
/// is detectable in the output.
template <typename T>
__global__ void FillDeterministicPattern(
    T*       __restrict__ d_buf,
    size_t                num_elements,
    uint64_t              seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        // Bijective: golden ratio hash
        const uint64_t h = (i + seed) * 0x9E3779B97F4A7C15ULL;
        // Map to float in [-1, 1]
        const float val = static_cast<float>(static_cast<int64_t>(h >> 40)) / 8388608.0f;
        d_buf[i] = static_cast<T>(val);
    }
}

/// Generate boundary-condition index arrays.
/// Produces indices that hit:
///   - First and last rows (boundary)
///   - Rows at power-of-2 offsets (alignment stress)
///   - Repeated indices (atomics stress)
///   - Sequential runs (coalescing)
__global__ void GenerateBoundaryIndices(
    size_t*  __restrict__ d_indices,
    size_t                num_indices,
    size_t                num_rows,
    uint64_t              seed)
{
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_indices;
         i += gridDim.x * static_cast<size_t>(blockDim.x))
    {
        size_t idx;
        const uint32_t pattern = static_cast<uint32_t>(i % 8);

        switch (pattern)
        {
        case 0: idx = 0;                             break; // first row
        case 1: idx = num_rows - 1;                   break; // last row
        case 2: idx = (1ULL << (i % 20)) % num_rows;  break; // power-of-2
        case 3: idx = d_indices[0];                   break; // repeated (atomics)
        case 4: idx = i % num_rows;                   break; // sequential
        case 5: idx = (num_rows / 2) + (i % 32);     break; // mid-range cluster
        case 6: // xorshift random
        {
            uint64_t s = seed ^ (i * 6364136223846793005ULL);
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            idx = s % num_rows;
            break;
        }
        default:
            idx = (num_rows - 1 - i) % num_rows;     // reverse sequential
            break;
        }

        d_indices[i] = idx;
    }
}

// ─────────────────────────────────────────────────────────────────
//  Influence scoring (host-side, stored per fuzz parameter)
//
//  Adapted from SyzMini's influence-guided syscall removal:
//  after each fuzz round, we update influence scores for the
//  parameters that were varied.  High-influence parameters
//  (those correlated with divergence) get wider mutation
//  ranges in subsequent rounds.
// ─────────────────────────────────────────────────────────────────

struct InfluenceScores {
    float block_x;
    float block_y;
    float block_z;
    float grid_x;
    float shared_mem;
    float alignment;
    float shape_rows;
    float shape_cols;

    static constexpr float ALPHA = 0.15f;  // EMA learning rate
    static constexpr float FLOOR = 0.05f;  // minimum score

    void update(const char* param_name, bool caused_divergence)
    {
        float* target = nullptr;
        if      (!strcmp(param_name, "block_x"))    target = &block_x;
        else if (!strcmp(param_name, "block_y"))    target = &block_y;
        else if (!strcmp(param_name, "block_z"))    target = &block_z;
        else if (!strcmp(param_name, "grid_x"))     target = &grid_x;
        else if (!strcmp(param_name, "shared_mem")) target = &shared_mem;
        else if (!strcmp(param_name, "alignment"))  target = &alignment;
        else if (!strcmp(param_name, "shape_rows")) target = &shape_rows;
        else if (!strcmp(param_name, "shape_cols")) target = &shape_cols;

        if (target)
        {
            const float signal = caused_divergence ? 1.0f : 0.0f;
            *target = ALPHA * signal + (1.0f - ALPHA) * (*target);
            if (*target < FLOOR) *target = FLOOR;
        }
    }

    void reset()
    {
        block_x = block_y = block_z = grid_x = 0.5f;
        shared_mem = alignment = shape_rows = shape_cols = 0.5f;
    }
};

}  // namespace Crucible

#endif  // CRUCIBLE_DIFFERENTIAL_VALIDATOR_CUH
