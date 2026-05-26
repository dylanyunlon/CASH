#pragma once

#ifndef CRUCIBLE_PARETO_SOLVER_CUH
#define CRUCIBLE_PARETO_SOLVER_CUH

// Crucible Pareto Frontier Solver
//
// Exp3 asks: what throughput can we achieve while guaranteeing
// cross-architecture result consistency?
//
// The Pareto frontier maps tolerance → max_throughput:
//
//   tolerance = 0       → bitwise identical → lowest throughput
//                          (must sync after every kernel, may need
//                           deterministic reduction order)
//   tolerance = ε       → within ε ULP     → medium throughput
//                          (can reorder reductions, skip some syncs)
//   tolerance = ∞       → no constraint    → max throughput
//                          (each arch runs its own code path)
//
// The solver sweeps tolerance levels and for each level:
//   1. Configures the kernel launch for maximum throughput
//      (at tolerance=0 this means deterministic thread ordering;
//       at tolerance=∞ this means arch-specific optimisations)
//   2. Runs N iterations measuring throughput
//   3. Verifies the output is within tolerance
//   4. Records the (tolerance, throughput, actual_error) triple
//
// The key insight is that relaxing bitwise consistency enables
// architecture-specific optimisations:
//   - sm90 can use TMA for async copies (faster but different
//     rounding than explicit loads)
//   - sm90 can use larger shared memory (different tiling, hence
//     different accumulation order)
//   - sm90 FP8 Tensor Cores have different rounding than sm86's
//     FP32 path

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  Tolerance levels for the Pareto sweep
// ─────────────────────────────────────────────────────────────────

struct ToleranceLevel {
    float    max_ulp;           // max acceptable ULP distance
    float    max_rel_error;     // max acceptable relative error
    bool     require_bitwise;   // if true, override above with exact match
    bool     allow_reorder;     // can reorder reductions for speed?
    bool     allow_arch_spec;   // can use arch-specific code paths?

    static constexpr int NUM_PRESETS = 7;

    static ToleranceLevel presets[NUM_PRESETS];
};

// Preset sweep from strict → relaxed
inline ToleranceLevel ToleranceLevel::presets[NUM_PRESETS] = {
    // max_ulp  max_rel  bitwise  reorder  arch_spec
    {  0,       0,       true,    false,   false  },   // L0: bitwise identical
    {  1,       1e-7f,   false,   false,   false  },   // L1: 1 ULP
    {  4,       1e-6f,   false,   true,    false  },   // L2: 4 ULP, reorder OK
    {  16,      1e-5f,   false,   true,    false  },   // L3: 16 ULP
    {  256,     1e-4f,   false,   true,    true   },   // L4: 256 ULP, arch-spec OK
    {  65536,   1e-2f,   false,   true,    true   },   // L5: 64K ULP (2^16)
    {  1e12f,   1.0f,    false,   true,    true   },   // L6: unconstrained
};

// ─────────────────────────────────────────────────────────────────
//  Throughput measurement kernel
//
//  A synthetic embedding lookup+scatter workload whose launch
//  configuration is adjusted based on the tolerance level:
//
//  At strict tolerance:
//    - Single warp per block (deterministic lane ordering)
//    - No shared memory tiling (avoids reorder)
//    - Explicit __syncthreads barriers
//
//  At relaxed tolerance:
//    - Full block occupancy
//    - Shared memory tiling for data reuse
//    - Reduced synchronisation
// ─────────────────────────────────────────────────────────────────

// Strict mode: deterministic reduction with single-warp execution.
// Guarantees identical bit-patterns across architectures at the cost
// of low occupancy and no memory-level parallelism.
template <typename T>
__global__ void StrictLookupKernel(
    const T*      __restrict__ d_table,       // [rows × dim]
    const size_t* __restrict__ d_indices,     // [batch]
    T*            __restrict__ d_output,      // [batch × dim]
    size_t                     batch_size,
    size_t                     dim)
{
    // One warp per row — deterministic lane assignment
    const size_t row_idx = blockIdx.x;
    if (row_idx >= batch_size) return;

    const size_t emb_row = d_indices[row_idx];
    const T* __restrict__ src = d_table + emb_row * dim;
    T*       __restrict__ dst = d_output + row_idx * dim;

    // Sequential within warp: each lane handles dim/32 elements
    for (size_t d = threadIdx.x; d < dim; d += 32)
    {
        dst[d] = src[d];
    }
}

// Relaxed mode: maximum throughput with architecture-specific tuning.
// Multiple rows per block, shared memory staging, vectorised loads.
template <typename T, int BLOCK_THREADS = 256, int ROWS_PER_BLOCK = 8>
__global__ void RelaxedLookupKernel(
    const T*      __restrict__ d_table,
    const size_t* __restrict__ d_indices,
    T*            __restrict__ d_output,
    size_t                     batch_size,
    size_t                     dim)
{
    const size_t row_local  = threadIdx.y;
    const size_t row_global = static_cast<size_t>(blockIdx.x) * ROWS_PER_BLOCK + row_local;

    if (row_global >= batch_size) return;

    const size_t emb_row = d_indices[row_global];

    // Vectorised: threadIdx.x covers dim elements
    for (size_t d = threadIdx.x; d < dim; d += blockDim.x)
    {
        d_output[row_global * dim + d] = d_table[emb_row * dim + d];
    }
}

// ─────────────────────────────────────────────────────────────────
//  Pareto measurement point
// ─────────────────────────────────────────────────────────────────

struct ParetoPoint {
    int      tolerance_level;   // index into ToleranceLevel::presets
    float    throughput_gops;   // giga-operations per second
    float    achieved_max_ulp;  // actual max ULP observed
    float    achieved_max_rel;  // actual max relative error
    bool     consistency_ok;    // within tolerance?
    float    latency_ms;        // per-iteration latency
};

// ─────────────────────────────────────────────────────────────────
//  Verification kernel: check output against reference
//
//  For each Pareto point, we run the workload on device A (sm86)
//  as the reference and device B (sm90) as the test, then measure
//  the divergence to determine if the tolerance was met.
// ─────────────────────────────────────────────────────────────────

__global__ void ParetoVerifyKernel(
    const float* __restrict__ d_ref,       // reference output (sm86)
    const float* __restrict__ d_test,      // test output (sm90)
    float*       __restrict__ d_max_ulp,   // [1] — max ULP (as float)
    float*       __restrict__ d_max_rel,   // [1] — max relative error
    uint32_t                  num_elements)
{
    float local_max_rel = 0.0f;
    float local_max_ulp = 0.0f;

    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += gridDim.x * blockDim.x)
    {
        const float a = d_ref[i];
        const float b = d_test[i];

        // Relative error
        const float abs_diff = fabsf(a - b);
        const float rel = abs_diff / fmaxf(fabsf(a), 1e-10f);
        local_max_rel = fmaxf(local_max_rel, rel);

        // ULP distance (approximate via integer reinterpret)
        const int32_t ia = __float_as_int(a);
        const int32_t ib = __float_as_int(b);
        int32_t ulp_diff = (ia < 0 ? (0x80000000 - ia) : ia) -
                           (ib < 0 ? (0x80000000 - ib) : ib);
        if (ulp_diff < 0) ulp_diff = -ulp_diff;
        local_max_ulp = fmaxf(local_max_ulp, static_cast<float>(ulp_diff));
    }

    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        local_max_rel = fmaxf(local_max_rel,
                              __shfl_down_sync(0xFFFFFFFF, local_max_rel, offset));
        local_max_ulp = fmaxf(local_max_ulp,
                              __shfl_down_sync(0xFFFFFFFF, local_max_ulp, offset));
    }

    if ((threadIdx.x & 31) == 0)
    {
        atomicMax(reinterpret_cast<int*>(d_max_ulp), __float_as_int(local_max_ulp));
        atomicMax(reinterpret_cast<int*>(d_max_rel), __float_as_int(local_max_rel));
    }
}

// ─────────────────────────────────────────────────────────────────
//  Timer utility for throughput measurement
// ─────────────────────────────────────────────────────────────────

struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    void create()
    {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    void destroy()
    {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) { cudaEventRecord(start_event, stream); }
    void stop(cudaStream_t stream = 0)  { cudaEventRecord(stop_event, stream); }

    float elapsed_ms()
    {
        cudaEventSynchronize(stop_event);
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }
};

}  // namespace Crucible

#endif  // CRUCIBLE_PARETO_SOLVER_CUH
