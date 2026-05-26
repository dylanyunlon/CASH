#pragma once

#ifndef CRUCIBLE_ARCH_DISPATCH_CUH
#define CRUCIBLE_ARCH_DISPATCH_CUH

/// Crucible Architecture-Aware Kernel Dispatch
///
/// The same CUDA kernel compiled for sm86 (A6000) and sm90 (H100)
/// can produce different results due to:
///
///   1. Different shared memory limits (48KB vs 228KB)
///   2. Different warp scheduling policies
///   3. FP8 hardware only on sm90
///   4. Asynchronous copy semantics (cp.async.bulk on sm90)
///   5. Thread Block Cluster (sm90 only)
///
/// This header provides:
///   - Compile-time architecture traits
///   - Runtime architecture detection
///   - Arch-dependent kernel configuration
///   - Shared memory limit enforcement

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <cassert>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  Architecture traits (compile-time)
// ─────────────────────────────────────────────────────────────────

template <int SM_MAJOR, int SM_MINOR>
struct ArchTraits;

// Ampere — sm86 (RTX A6000)
template <>
struct ArchTraits<8, 6> {
    static constexpr int  sm_version               = 86;
    static constexpr int  max_threads_per_block     = 1024;
    static constexpr int  max_shared_mem_per_block   = 49152;   // 48 KB
    static constexpr int  max_shared_mem_optin       = 99 * 1024; // 99 KB w/ opt-in
    static constexpr int  max_registers_per_block    = 65536;
    static constexpr int  warp_size                  = 32;
    static constexpr int  sm_count_typical           = 84;      // A6000
    static constexpr bool has_fp8                    = false;
    static constexpr bool has_tma                    = false;    // no TMA
    static constexpr bool has_cluster                = false;
    static constexpr int  pcie_gen                   = 4;
    static constexpr float memory_bandwidth_gbps     = 768.0f;  // GDDR6X
};

// Hopper — sm90 (H100)
template <>
struct ArchTraits<9, 0> {
    static constexpr int  sm_version               = 90;
    static constexpr int  max_threads_per_block     = 1024;
    static constexpr int  max_shared_mem_per_block   = 49152;   // 48 KB default
    static constexpr int  max_shared_mem_optin       = 228 * 1024; // 228 KB w/ opt-in
    static constexpr int  max_registers_per_block    = 65536;
    static constexpr int  warp_size                  = 32;
    static constexpr int  sm_count_typical           = 132;     // H100 SXM5
    static constexpr bool has_fp8                    = true;
    static constexpr bool has_tma                    = true;     // Tensor Memory Accelerator
    static constexpr bool has_cluster                = true;     // Thread Block Clusters
    static constexpr int  pcie_gen                   = 5;
    static constexpr float memory_bandwidth_gbps     = 3350.0f; // HBM3
};

using SM86 = ArchTraits<8, 6>;
using SM90 = ArchTraits<9, 0>;

// ─────────────────────────────────────────────────────────────────
//  Runtime architecture detection
// ─────────────────────────────────────────────────────────────────

struct DeviceArch {
    int device_id;
    int sm_major;
    int sm_minor;
    int sm_count;
    int max_shared_mem;
    int max_shared_mem_optin;
    int max_threads_per_block;
    int max_registers_per_block;
    int pcie_gen;         // Detected or estimated
    float memory_bw_gbps; // From cudaDeviceProp

    static DeviceArch detect(int device_id)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);

        DeviceArch arch;
        arch.device_id              = device_id;
        arch.sm_major               = prop.major;
        arch.sm_minor               = prop.minor;
        arch.sm_count               = prop.multiProcessorCount;
        arch.max_shared_mem         = static_cast<int>(prop.sharedMemPerBlock);
        arch.max_shared_mem_optin   = static_cast<int>(prop.sharedMemPerBlockOptin);
        arch.max_threads_per_block  = prop.maxThreadsPerBlock;
        arch.max_registers_per_block = prop.regsPerBlock;
        arch.pcie_gen = (prop.major >= 9) ? 5 : 4;
        // Approximate bandwidth from clock and bus width
        arch.memory_bw_gbps = static_cast<float>(
            2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);

        return arch;
    }

    bool is_hopper()  const { return sm_major >= 9; }
    bool is_ampere()  const { return sm_major == 8; }
    bool has_fp8()    const { return sm_major >= 9; }
    int  sm_version() const { return sm_major * 10 + sm_minor; }
};

// ─────────────────────────────────────────────────────────────────
//  Kernel launch configuration that respects arch limits
//
//  The key insight: a configuration that is valid on sm90 may
//  silently corrupt on sm86 if it requests more shared memory
//  than 48KB (default) or uses grid dimensions beyond the arch's
//  maximum.  Crucible enforces these limits before launch.
// ─────────────────────────────────────────────────────────────────

struct KernelLaunchConfig {
    dim3     grid;
    dim3     block;
    size_t   shared_mem_bytes;
    int      target_device;

    enum class Validity {
        OK,
        SHARED_MEM_EXCEEDED,
        THREADS_EXCEEDED,
        GRID_EXCEEDED,
        DEVICE_MISMATCH
    };

    /// Validate this config against a detected device architecture.
    Validity validate(const DeviceArch& arch) const
    {
        const int total_threads = block.x * block.y * block.z;
        if (total_threads > arch.max_threads_per_block)
            return Validity::THREADS_EXCEEDED;

        if (static_cast<int>(shared_mem_bytes) > arch.max_shared_mem_optin)
            return Validity::SHARED_MEM_EXCEEDED;

        // sm86 default shared mem limit is 48KB; requesting more without
        // cudaFuncSetAttribute will silently fall back or fail
        if (!arch.is_hopper() &&
            static_cast<int>(shared_mem_bytes) > 49152)
            return Validity::SHARED_MEM_EXCEEDED;

        return Validity::OK;
    }

    /// Clamp this config to be valid on the given architecture.
    /// Returns true if clamping was needed (i.e., original was invalid).
    bool clamp_to_arch(const DeviceArch& arch)
    {
        bool clamped = false;

        // Clamp block dimensions
        int total = block.x * block.y * block.z;
        while (total > arch.max_threads_per_block && block.x > 32)
        {
            block.x /= 2;
            total = block.x * block.y * block.z;
            clamped = true;
        }

        // Clamp shared memory
        const int limit = arch.is_hopper()
            ? arch.max_shared_mem_optin
            : arch.max_shared_mem;

        if (static_cast<int>(shared_mem_bytes) > limit)
        {
            shared_mem_bytes = static_cast<size_t>(limit);
            clamped = true;
        }

        return clamped;
    }
};

// ─────────────────────────────────────────────────────────────────
//  Boundary value generator for kernel parameter fuzzing
//
//  Generates launch configurations at architecture-relevant
//  boundaries — the exact points where sm86 and sm90 diverge.
//  Used by the differential fuzzer to find silent corruptions.
// ─────────────────────────────────────────────────────────────────

struct BoundaryConfig {
    static constexpr int SHARED_MEM_BOUNDARIES[] = {
        0,                  // no shared mem
        1024,               // minimal
        48 * 1024,          // sm86 default max
        48 * 1024 + 1,      // just over sm86 default → fails on sm86 without opt-in
        64 * 1024,          // common opt-in point
        99 * 1024,          // sm86 opt-in max
        99 * 1024 + 1,      // just over sm86 opt-in → fails on sm86 always
        164 * 1024,         // sm90 mid-range
        228 * 1024,         // sm90 opt-in max
    };

    static constexpr int BLOCK_SIZES[] = {
        32, 64, 128, 256, 512, 1024
    };

    static constexpr int ALIGNMENT_VALUES[] = {
        1, 2, 4, 8, 16, 32, 64, 128, 256
    };

    static constexpr int NUM_SHARED_MEM = sizeof(SHARED_MEM_BOUNDARIES) / sizeof(int);
    static constexpr int NUM_BLOCKS     = sizeof(BLOCK_SIZES) / sizeof(int);
    static constexpr int NUM_ALIGN      = sizeof(ALIGNMENT_VALUES) / sizeof(int);
};

constexpr int BoundaryConfig::SHARED_MEM_BOUNDARIES[];
constexpr int BoundaryConfig::BLOCK_SIZES[];
constexpr int BoundaryConfig::ALIGNMENT_VALUES[];

}  // namespace Crucible

#endif  // CRUCIBLE_ARCH_DISPATCH_CUH
