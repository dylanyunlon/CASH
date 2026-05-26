#pragma once

#ifndef CRUCIBLE_WORKLOAD_PARTITIONER_CUH
#define CRUCIBLE_WORKLOAD_PARTITIONER_CUH

// Crucible Workload Partitioner
//
// Adapts PIM-ANNS's per-core fine-grained scheduling to heterogeneous
// GPU clusters.  The fundamental problem:
//
//   Given a mixed workload W = {embedding_lookup, gradient_update},
//   and devices D = {H100 (fast, few), A6000×2 (slower, more)},
//   find assignment A: W → D that minimises max_d∈D T(A, d).
//
// This is a weighted load-balancing problem where device throughput
// is workload-dependent (H100's HBM3 advantage is larger for
// bandwidth-bound lookups than compute-bound gradient updates).
//
// The partitioner:
//   1. Calibrates each device with micro-benchmarks (Calibration kernel)
//   2. Builds a per-device cost model
//   3. Solves the partition via a greedy bisection on the objective
//
// The calibration kernel measures two metrics per device:
//   - Embedding lookup throughput (random-access reads)
//   - Gradient scatter throughput (random-access writes + atomics)
//
// These are measured in-situ because they depend on the actual
// memory system behaviour (cache hierarchy, TLB, bank conflicts),
// not just peak bandwidth numbers from the spec sheet.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  Calibration kernel: random-access read throughput
//
//  Simulates embedding lookup by performing gather operations
//  with Zipf-distributed indices (heavy-tailed access pattern
//  that stresses the cache hierarchy).
// ─────────────────────────────────────────────────────────────────

template <typename T, int BLOCK_THREADS = 256, int ITERS = 64>
__global__ void CalibrateLookupKernel(
    const T*      __restrict__ d_table,      // [num_rows × dim]
    const size_t* __restrict__ d_indices,    // [num_probes]
    T*            __restrict__ d_output,     // [num_probes × dim]
    size_t                     num_probes,
    size_t                     dim,
    float*        __restrict__ d_elapsed_ms) // [1] — output
{
    // We measure wall-clock time using clock64() difference.
    // Only thread 0 of block 0 records the timer; all threads
    // participate in the actual work to get realistic throughput.

    long long start_clock = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        start_clock = clock64();

    __syncthreads();

    // Each thread handles multiple probes (grid-stride loop)
    for (size_t p = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         p < num_probes;
         p += gridDim.x * BLOCK_THREADS)
    {
        const size_t row = d_indices[p];
        const T* __restrict__ src = d_table + row * dim;
        T*       __restrict__ dst = d_output + p * dim;

        // Gather one embedding row
        for (size_t d = 0; d < dim; ++d)
        {
            dst[d] = src[d];
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        const long long end_clock = clock64();
        // Convert clock cycles to approximate milliseconds
        // (device clock rate in kHz is available from cudaDeviceProp)
        *d_elapsed_ms = static_cast<float>(end_clock - start_clock);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Calibration kernel: random-access write (scatter) throughput
//
//  Simulates gradient update by performing atomicAdd scatter
//  operations — the write side of embedding training.
// ─────────────────────────────────────────────────────────────────

template <int BLOCK_THREADS = 256>
__global__ void CalibrateScatterKernel(
    float*        __restrict__ d_grad_table,  // [num_rows × dim]
    const size_t* __restrict__ d_indices,     // [num_probes]
    const float*  __restrict__ d_values,      // [num_probes × dim]
    size_t                     num_probes,
    size_t                     dim,
    float*        __restrict__ d_elapsed_ms)
{
    long long start_clock = 0;
    if (threadIdx.x == 0 && blockIdx.x == 0)
        start_clock = clock64();

    __syncthreads();

    for (size_t p = blockIdx.x * BLOCK_THREADS + threadIdx.x;
         p < num_probes;
         p += gridDim.x * BLOCK_THREADS)
    {
        const size_t row = d_indices[p];
        for (size_t d = 0; d < dim; ++d)
        {
            atomicAdd(d_grad_table + row * dim + d, d_values[p * dim + d]);
        }
    }

    __syncthreads();

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *d_elapsed_ms = static_cast<float>(clock64() - start_clock);
    }
}

// ─────────────────────────────────────────────────────────────────
//  Device performance profile (populated by calibration)
// ─────────────────────────────────────────────────────────────────

struct DeviceProfile {
    int    device_id;
    int    sm_version;
    int    sm_count;
    float  clock_rate_khz;

    // Measured throughput (operations per millisecond)
    float  lookup_ops_per_ms;
    float  scatter_ops_per_ms;

    // Derived: cost per operation (inverse throughput)
    float  lookup_cost_per_op()  const { return 1.0f / lookup_ops_per_ms; }
    float  scatter_cost_per_op() const { return 1.0f / scatter_ops_per_ms; }
};

// ─────────────────────────────────────────────────────────────────
//  Workload partition plan
// ─────────────────────────────────────────────────────────────────

struct PartitionPlan {
    static constexpr int MAX_DEVICES = 8;

    int    num_devices;
    int    device_ids[MAX_DEVICES];
    float  fractions[MAX_DEVICES];      // fraction of workload assigned
    float  estimated_time_ms[MAX_DEVICES]; // per-device estimated time
    float  total_time_ms;               // max across devices (= makespan)
    float  load_balance;                // min_time / max_time ∈ (0, 1]
    int    bottleneck_device;
};

// ─────────────────────────────────────────────────────────────────
//  Partition solver
//
//  Given N device profiles and a workload (num_lookups + num_scatters),
//  finds the fraction assignment that minimises makespan.
//
//  Strategy: proportional assignment weighted by device throughput.
//  For the two-operation case (lookup + scatter), the optimal
//  fraction for device i is:
//
//    f_i = throughput_i / Σ throughput_j
//
//  where throughput_i is the harmonic mean of lookup and scatter
//  throughput (since both operations must complete on each device,
//  the bottleneck operation determines device time).
//
//  This is computed on the host at the start of each training
//  step — the overhead is negligible (~100ns for 3 devices).
// ─────────────────────────────────────────────────────────────────

inline PartitionPlan solve_partition(
    const DeviceProfile* profiles,
    int                  num_devices,
    size_t               num_lookups,
    size_t               num_scatters)
{
    PartitionPlan plan;
    plan.num_devices = num_devices;

    // Compute per-device effective throughput (harmonic mean)
    float total_throughput = 0.0f;
    float device_throughput[PartitionPlan::MAX_DEVICES];

    for (int i = 0; i < num_devices; ++i)
    {
        plan.device_ids[i] = profiles[i].device_id;

        // Harmonic mean of lookup and scatter throughput
        const float l = profiles[i].lookup_ops_per_ms;
        const float s = profiles[i].scatter_ops_per_ms;
        device_throughput[i] = (l > 0 && s > 0)
            ? 2.0f * l * s / (l + s)
            : 0.0f;
        total_throughput += device_throughput[i];
    }

    // Assign fractions proportional to throughput
    float max_time = 0.0f;
    float min_time = 1e30f;

    for (int i = 0; i < num_devices; ++i)
    {
        plan.fractions[i] = (total_throughput > 0)
            ? device_throughput[i] / total_throughput
            : 1.0f / num_devices;

        // Estimated time for this device's share
        const float ops_lookup  = num_lookups  * plan.fractions[i];
        const float ops_scatter = num_scatters * plan.fractions[i];
        const float t_lookup    = ops_lookup  * profiles[i].lookup_cost_per_op();
        const float t_scatter   = ops_scatter * profiles[i].scatter_cost_per_op();
        plan.estimated_time_ms[i] = t_lookup + t_scatter;

        if (plan.estimated_time_ms[i] > max_time)
        {
            max_time = plan.estimated_time_ms[i];
            plan.bottleneck_device = i;
        }
        min_time = (plan.estimated_time_ms[i] < min_time)
            ? plan.estimated_time_ms[i]
            : min_time;
    }

    plan.total_time_ms = max_time;
    plan.load_balance  = (max_time > 0) ? min_time / max_time : 1.0f;

    return plan;
}

// Uniform partition (baseline comparison)
inline PartitionPlan solve_uniform(
    const DeviceProfile* profiles,
    int                  num_devices,
    size_t               num_lookups,
    size_t               num_scatters)
{
    PartitionPlan plan;
    plan.num_devices = num_devices;
    const float frac = 1.0f / num_devices;

    float max_time = 0.0f;
    float min_time = 1e30f;

    for (int i = 0; i < num_devices; ++i)
    {
        plan.device_ids[i] = profiles[i].device_id;
        plan.fractions[i]  = frac;

        const float t_lookup  = num_lookups  * frac * profiles[i].lookup_cost_per_op();
        const float t_scatter = num_scatters * frac * profiles[i].scatter_cost_per_op();
        plan.estimated_time_ms[i] = t_lookup + t_scatter;

        if (plan.estimated_time_ms[i] > max_time)
        {
            max_time = plan.estimated_time_ms[i];
            plan.bottleneck_device = i;
        }
        min_time = (plan.estimated_time_ms[i] < min_time)
            ? plan.estimated_time_ms[i] : min_time;
    }

    plan.total_time_ms = max_time;
    plan.load_balance  = (max_time > 0) ? min_time / max_time : 1.0f;

    return plan;
}

}  // namespace Crucible

#endif  // CRUCIBLE_WORKLOAD_PARTITIONER_CUH
