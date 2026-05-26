#include "crucible/arch_dispatch.cuh"
#include "crucible/differential_validator.cuh"
#include "crucible/workload_partitioner.cuh"

#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  CrucibleRuntime
//
//  Orchestrates the differential testing and workload scheduling
//  pipeline for a heterogeneous GPU cluster.
//
//  Lifecycle:
//    1. detect()       — enumerate GPUs, classify into sm86/sm90
//    2. calibrate()    — run micro-benchmarks on each device
//    3. fuzz_round()   — one round of differential fuzzing
//    4. partition()    — compute optimal workload split
//    5. verify()       — run correctness check with current partition
// ─────────────────────────────────────────────────────────────────

class CrucibleRuntime {
public:
    CrucibleRuntime() { memset(&influence_, 0, sizeof(influence_)); }

    /// Detect all GPUs and classify by architecture.
    void detect()
    {
        int num_devices = 0;
        cudaGetDeviceCount(&num_devices);

        for (int i = 0; i < num_devices; ++i)
        {
            DeviceArch arch = DeviceArch::detect(i);
            devices_.push_back(arch);

            if (arch.is_hopper())
                sm90_devices_.push_back(i);
            else if (arch.is_ampere())
                sm86_devices_.push_back(i);

            printf("[Crucible] GPU %d: sm%d%d (%d SMs, shmem=%dKB/%dKB, PCIe gen%d)\n",
                   i, arch.sm_major, arch.sm_minor, arch.sm_count,
                   arch.max_shared_mem / 1024,
                   arch.max_shared_mem_optin / 1024,
                   arch.pcie_gen);
        }

        printf("[Crucible] Cluster: %zu sm86 (A6000) + %zu sm90 (H100)\n",
               sm86_devices_.size(), sm90_devices_.size());

        influence_.reset();
    }

    /// Calibrate each device with lookup and scatter micro-benchmarks.
    void calibrate(size_t num_rows = 100000, size_t dim = 128,
                   size_t num_probes = 50000)
    {
        profiles_.clear();

        for (const auto& arch : devices_)
        {
            cudaSetDevice(arch.device_id);
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, arch.device_id);

            DeviceProfile prof;
            prof.device_id  = arch.device_id;
            prof.sm_version = arch.sm_version();
            prof.sm_count   = arch.sm_count;
            prof.clock_rate_khz = static_cast<float>(prop.clockRate);

            // ── Allocate calibration buffers ──
            float* d_table   = nullptr;
            float* d_output  = nullptr;
            float* d_values  = nullptr;
            size_t* d_indices = nullptr;
            float* d_elapsed = nullptr;

            const size_t table_bytes  = num_rows * dim * sizeof(float);
            const size_t probe_bytes  = num_probes * dim * sizeof(float);
            const size_t idx_bytes    = num_probes * sizeof(size_t);

            cudaMalloc(&d_table,   table_bytes);
            cudaMalloc(&d_output,  probe_bytes);
            cudaMalloc(&d_values,  probe_bytes);
            cudaMalloc(&d_indices, idx_bytes);
            cudaMalloc(&d_elapsed, sizeof(float));

            // Fill with deterministic data
            const int fill_grid = static_cast<int>((num_rows * dim + 255) / 256);
            FillDeterministicPattern<float><<<fill_grid, 256>>>(
                d_table, num_rows * dim, 42);
            FillDeterministicPattern<float><<<fill_grid, 256>>>(
                d_values, num_probes * dim, 137);
            GenerateBoundaryIndices<<<(num_probes + 255) / 256, 256>>>(
                d_indices, num_probes, num_rows, 7);
            cudaDeviceSynchronize();

            // ── Measure lookup throughput ──
            cudaMemset(d_elapsed, 0, sizeof(float));
            const int grid = static_cast<int>((num_probes + 255) / 256);
            CalibrateLookupKernel<float, 256><<<grid, 256>>>(
                d_table, d_indices, d_output, num_probes, dim, d_elapsed);
            cudaDeviceSynchronize();

            float elapsed_clocks = 0;
            cudaMemcpy(&elapsed_clocks, d_elapsed, sizeof(float), cudaMemcpyDeviceToHost);
            float elapsed_ms = elapsed_clocks / prof.clock_rate_khz;
            prof.lookup_ops_per_ms = (elapsed_ms > 0)
                ? static_cast<float>(num_probes) / elapsed_ms
                : 1.0f;

            // ── Measure scatter throughput ──
            cudaMemset(d_elapsed, 0, sizeof(float));
            CalibrateScatterKernel<256><<<grid, 256>>>(
                d_table, d_indices, d_values, num_probes, dim, d_elapsed);
            cudaDeviceSynchronize();

            cudaMemcpy(&elapsed_clocks, d_elapsed, sizeof(float), cudaMemcpyDeviceToHost);
            elapsed_ms = elapsed_clocks / prof.clock_rate_khz;
            prof.scatter_ops_per_ms = (elapsed_ms > 0)
                ? static_cast<float>(num_probes) / elapsed_ms
                : 1.0f;

            printf("[Crucible] GPU %d calibrated: lookup=%.0f ops/ms, scatter=%.0f ops/ms\n",
                   arch.device_id, prof.lookup_ops_per_ms, prof.scatter_ops_per_ms);

            profiles_.push_back(prof);

            cudaFree(d_table);
            cudaFree(d_output);
            cudaFree(d_values);
            cudaFree(d_indices);
            cudaFree(d_elapsed);
        }
    }

    /// Run one differential fuzzing round.
    /// Tests a kernel wrapper on all sm86/sm90 device pairs.
    /// Returns number of divergences found.
    template <typename KernelFunc>
    int fuzz_round(
        KernelFunc           kernel_fn,
        size_t               num_rows,
        size_t               dim,
        size_t               num_configs = 50)
    {
        if (sm86_devices_.empty() || sm90_devices_.empty())
        {
            printf("[Crucible] Need at least one sm86 + one sm90 for differential fuzzing\n");
            return 0;
        }

        const int dev_a = sm86_devices_[0];
        const int dev_b = sm90_devices_[0];
        int divergences = 0;

        // Generate boundary configurations
        for (size_t ci = 0; ci < num_configs; ++ci)
        {
            const int smem_idx  = ci % BoundaryConfig::NUM_SHARED_MEM;
            const int block_idx = (ci / BoundaryConfig::NUM_SHARED_MEM) % BoundaryConfig::NUM_BLOCKS;
            const int shared_mem = BoundaryConfig::SHARED_MEM_BOUNDARIES[smem_idx];
            const int block_size = BoundaryConfig::BLOCK_SIZES[block_idx];

            KernelLaunchConfig config;
            config.grid  = dim3(static_cast<unsigned>((num_rows + block_size - 1) / block_size));
            config.block = dim3(static_cast<unsigned>(block_size));
            config.shared_mem_bytes = static_cast<size_t>(shared_mem);

            // Check validity on both architectures
            const DeviceArch arch_a = DeviceArch::detect(dev_a);
            const DeviceArch arch_b = DeviceArch::detect(dev_b);

            auto valid_a = config.validate(arch_a);
            auto valid_b = config.validate(arch_b);

            // Interesting case: valid on sm90 but not sm86
            bool interesting = (valid_b == KernelLaunchConfig::Validity::OK &&
                                valid_a != KernelLaunchConfig::Validity::OK);

            if (valid_a != KernelLaunchConfig::Validity::OK &&
                valid_b != KernelLaunchConfig::Validity::OK)
                continue;  // Invalid on both — skip

            // Clamp for sm86 and run on both
            KernelLaunchConfig config_a = config;
            config_a.clamp_to_arch(arch_a);

            // Allocate output buffers
            const size_t out_bytes = num_rows * dim * sizeof(float);
            float* d_out_a = nullptr;
            float* d_out_b = nullptr;

            cudaSetDevice(dev_a);
            cudaMalloc(&d_out_a, out_bytes);
            cudaSetDevice(dev_b);
            cudaMalloc(&d_out_b, out_bytes);

            // Run kernel on both devices
            kernel_fn(dev_a, config_a, d_out_a, num_rows, dim);
            kernel_fn(dev_b, config,   d_out_b, num_rows, dim);

            // Copy sm90 output to sm86 device for comparison
            float* d_out_b_copy = nullptr;
            cudaSetDevice(dev_a);
            cudaMalloc(&d_out_b_copy, out_bytes);
            cudaMemcpy(d_out_b_copy, d_out_b, out_bytes, cudaMemcpyDeviceToDevice);

            // Compare
            DivergenceReport* d_report = nullptr;
            cudaMalloc(&d_report, sizeof(DivergenceReport));
            cudaMemset(d_report, 0, sizeof(DivergenceReport));

            const int cmp_grid = static_cast<int>((num_rows * dim + 255) / 256);
            BitwiseCompareKernel<float, 256><<<cmp_grid, 256>>>(
                d_out_a, d_out_b_copy,
                static_cast<uint32_t>(num_rows * dim),
                d_report);

            DivergenceReport report;
            cudaMemcpy(&report, d_report, sizeof(DivergenceReport), cudaMemcpyDeviceToHost);

            if (report.mismatched_elements > 0)
            {
                ++divergences;
                printf("[Crucible] DIVERGENCE config=%zu: block=%d shmem=%dKB "
                       "mismatches=%u max_abs=%.6e max_rel=%.6e%s\n",
                       ci, block_size, shared_mem / 1024,
                       report.mismatched_elements,
                       report.max_abs_diff,
                       report.max_rel_diff,
                       interesting ? " [arch-boundary]" : "");
            }

            // Update influence scores
            influence_.update("shared_mem", report.mismatched_elements > 0);
            influence_.update("block_x",    report.mismatched_elements > 0);

            cudaFree(d_out_a);
            cudaFree(d_out_b);
            cudaFree(d_out_b_copy);
            cudaFree(d_report);
        }

        printf("[Crucible] Fuzz round: %d/%zu divergences (%.1f%%)\n",
               divergences, num_configs,
               100.0f * divergences / num_configs);
        printf("[Crucible] Influence: shmem=%.3f block_x=%.3f align=%.3f\n",
               influence_.shared_mem, influence_.block_x, influence_.alignment);

        return divergences;
    }

    /// Compute optimal workload partition.
    PartitionPlan partition(size_t num_lookups, size_t num_scatters) const
    {
        return solve_partition(profiles_.data(),
                              static_cast<int>(profiles_.size()),
                              num_lookups, num_scatters);
    }

    /// Compute uniform partition (for comparison).
    PartitionPlan partition_uniform(size_t num_lookups, size_t num_scatters) const
    {
        return solve_uniform(profiles_.data(),
                             static_cast<int>(profiles_.size()),
                             num_lookups, num_scatters);
    }

    // ── Accessors ───────────────────────────────────────────
    const std::vector<DeviceArch>&   devices()  const { return devices_; }
    const std::vector<DeviceProfile>& profiles() const { return profiles_; }
    const InfluenceScores& influence()           const { return influence_; }

private:
    std::vector<DeviceArch>    devices_;
    std::vector<int>           sm86_devices_;
    std::vector<int>           sm90_devices_;
    std::vector<DeviceProfile> profiles_;
    InfluenceScores            influence_;
};

}  // namespace Crucible
