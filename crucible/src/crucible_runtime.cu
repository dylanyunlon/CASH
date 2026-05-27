#include "crucible/arch_dispatch.cuh"
#include "crucible/differential_validator.cuh"
#include "crucible/workload_partitioner.cuh"
#include "crucible/ulp_analyzer.cuh"
#include "crucible/pareto_solver.cuh"
#include "crucible/index_kernel_harness.cuh"

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

    // Detect all GPUs and classify by architecture.
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

    // Calibrate each device with lookup and scatter micro-benchmarks.
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

    // Run differential fuzzing over HypeReca's three index kernels
    // using BoundaryTestSuite configurations and IndexKernelHarness.
    // Returns total divergences found across all kernels × configs.
    int fuzz_round(size_t max_configs = 100)
    {
        if (sm86_devices_.empty() || sm90_devices_.empty())
        {
            printf("[Crucible] Need at least one sm86 + one sm90\n");
            return 0;
        }

        const int dev_a = sm86_devices_[0];
        const int dev_b = sm90_devices_[0];
        int total_divergences = 0;

        // Generate boundary configs via BoundaryTestSuite
        std::vector<IndexKernelTestConfig> configs(max_configs);
        const size_t num_configs = BoundaryTestSuite::generate(
            configs.data(), max_configs);

        // Three kernels to test — same set Alloy's TieredEmbedding depends on
        using KT = IndexKernelHarness::KernelType;
        const struct { KT type; const char* name; } kernels[] = {
            {KT::GET,  "indexGet"},
            {KT::PUT,  "indexPut"},
            {KT::COPY, "indexCopy"},
        };

        for (const auto& kern : kernels)
        {
            int kern_div = 0;

            for (size_t ci = 0; ci < num_configs; ++ci)
            {
                const auto& cfg = configs[ci];

                // Skip if embedding_dim exceeds block limit
                if (cfg.embedding_dim > 1024) continue;

                DivergenceReport report =
                    IndexKernelHarness::differential_test<float>(
                        dev_a, dev_b, kern.type, cfg);

                const bool diverged = report.mismatched_elements > 0;

                if (diverged)
                {
                    ++kern_div;
                    printf("[Crucible] %s DIVERGENCE: dim=%zu rows=%zu "
                           "mismatches=%u max_abs=%.6e\n",
                           kern.name, cfg.embedding_dim, cfg.num_rows,
                           report.mismatched_elements, report.max_abs_diff);
                }

                // Update influence scores per parameter
                influence_.update("shape_rows", diverged);
                influence_.update("shape_cols", diverged);
                influence_.update("alignment", diverged);
            }

            printf("[Crucible] %s: %d/%zu divergences\n",
                   kern.name, kern_div, num_configs);
            total_divergences += kern_div;
        }

        printf("[Crucible] Total: %d divergences across %zu configs × 3 kernels\n",
               total_divergences, num_configs);
        printf("[Crucible] Influence: rows=%.3f cols=%.3f align=%.3f\n",
               influence_.shape_rows, influence_.shape_cols, influence_.alignment);

        return total_divergences;
    }

    // Compute optimal workload partition.
    PartitionPlan partition(size_t num_lookups, size_t num_scatters) const
    {
        return solve_partition(profiles_.data(),
                              static_cast<int>(profiles_.size()),
                              num_lookups, num_scatters);
    }

    // Compute uniform partition (for comparison).
    PartitionPlan partition_uniform(size_t num_lookups, size_t num_scatters) const
    {
        return solve_uniform(profiles_.data(),
                             static_cast<int>(profiles_.size()),
                             num_lookups, num_scatters);
    }

    // Sweep the Pareto frontier: for each tolerance level, measure
    // throughput on both architectures and verify error is within bound.
    // Returns one ParetoPoint per ToleranceLevel preset.
    std::vector<ParetoPoint> sweep_pareto(
        size_t num_rows = 10000,
        size_t dim      = 128,
        size_t batch    = 4096,
        int    iters    = 200)
    {
        if (sm86_devices_.empty() || sm90_devices_.empty())
        {
            printf("[Crucible] sweep_pareto requires at least one sm86 + one sm90\n");
            return {};
        }

        const int dev_a = sm86_devices_[0];
        const int dev_b = sm90_devices_[0];
        const size_t out_elems = batch * dim;
        const size_t table_bytes = num_rows * dim * sizeof(float);
        const size_t out_bytes   = out_elems * sizeof(float);
        const size_t idx_bytes   = batch * sizeof(size_t);

        std::vector<ParetoPoint> points;

        for (int lvl = 0; lvl < ToleranceLevel::NUM_PRESETS; ++lvl)
        {
            const auto& tol = ToleranceLevel::presets[lvl];

            // Allocate per-device
            float *d_table_a = nullptr, *d_table_b = nullptr;
            float *d_out_a = nullptr, *d_out_b = nullptr;
            size_t *d_idx_a = nullptr, *d_idx_b = nullptr;

            cudaSetDevice(dev_a);
            cudaMalloc(&d_table_a, table_bytes);
            cudaMalloc(&d_out_a, out_bytes);
            cudaMalloc(&d_idx_a, idx_bytes);
            FillDeterministicPattern<float><<<(num_rows*dim+255)/256, 256>>>(
                d_table_a, num_rows * dim, 42);
            GenerateBoundaryIndices<<<(batch+255)/256, 256>>>(
                d_idx_a, batch, num_rows, 7);

            cudaSetDevice(dev_b);
            cudaMalloc(&d_table_b, table_bytes);
            cudaMalloc(&d_out_b, out_bytes);
            cudaMalloc(&d_idx_b, idx_bytes);
            cudaMemcpy(d_table_b, d_table_a, table_bytes, cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_idx_b, d_idx_a, idx_bytes, cudaMemcpyDeviceToDevice);

            // Measure throughput on sm90 with the appropriate kernel
            GpuTimer timer;
            timer.create();

            cudaSetDevice(dev_b);
            cudaDeviceSynchronize();
            timer.start();

            for (int i = 0; i < iters; ++i)
            {
                if (tol.require_bitwise)
                {
                    StrictLookupKernel<float><<<batch, 32>>>(
                        d_table_b, d_idx_b, d_out_b, batch, dim);
                }
                else
                {
                    RelaxedLookupKernel<float, 256, 8>
                        <<<(batch+7)/8, dim3(32, 8)>>>(
                        d_table_b, d_idx_b, d_out_b, batch, dim);
                }
            }

            timer.stop();
            float total_ms = timer.elapsed_ms();

            // Run reference on sm86 (always strict for gold standard)
            cudaSetDevice(dev_a);
            for (int i = 0; i < iters; ++i)
            {
                StrictLookupKernel<float><<<batch, 32>>>(
                    d_table_a, d_idx_a, d_out_a, batch, dim);
            }
            cudaDeviceSynchronize();

            // Verify: compare sm86 reference vs sm90 output
            cudaSetDevice(dev_a);
            float *d_out_b_copy = nullptr;
            cudaMalloc(&d_out_b_copy, out_bytes);
            cudaMemcpy(d_out_b_copy, d_out_b, out_bytes, cudaMemcpyDeviceToDevice);

            // Use ULP-bounded comparison at this tolerance level
            float *d_max_ulp = nullptr, *d_max_rel = nullptr;
            cudaMalloc(&d_max_ulp, sizeof(float));
            cudaMalloc(&d_max_rel, sizeof(float));
            cudaMemset(d_max_ulp, 0, sizeof(float));
            cudaMemset(d_max_rel, 0, sizeof(float));

            ParetoVerifyKernel<<<(out_elems+255)/256, 256>>>(
                d_out_a, d_out_b_copy, d_max_ulp, d_max_rel,
                static_cast<uint32_t>(out_elems));

            float h_ulp = 0, h_rel = 0;
            cudaMemcpy(&h_ulp, d_max_ulp, sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_rel, d_max_rel, sizeof(float), cudaMemcpyDeviceToHost);

            float ops = static_cast<float>(batch) * dim * iters;
            ParetoPoint pt;
            pt.tolerance_level = lvl;
            pt.throughput_gops = ops / (total_ms * 1e6f);
            pt.achieved_max_ulp = h_ulp;
            pt.achieved_max_rel = h_rel;
            pt.consistency_ok   = tol.require_bitwise ? (h_ulp == 0) : (h_ulp <= tol.max_ulp);
            pt.latency_ms       = total_ms / iters;
            points.push_back(pt);

            printf("[Crucible] Pareto L%d: %.2f Gops/s  ulp=%.0f  rel=%.2e  %s\n",
                   lvl, pt.throughput_gops, pt.achieved_max_ulp,
                   pt.achieved_max_rel,
                   pt.consistency_ok ? "OK" : "VIOLATED");

            timer.destroy();
            cudaFree(d_table_a); cudaFree(d_out_a); cudaFree(d_idx_a);
            cudaFree(d_table_b); cudaFree(d_out_b); cudaFree(d_idx_b);
            cudaFree(d_out_b_copy); cudaFree(d_max_ulp); cudaFree(d_max_rel);
        }

        return points;
    }

    // Run ULP analysis on a fuzz result pair — produces a histogram
    // suitable as a paper figure.
    void analyze_ulp(
        float* d_output_a,
        float* d_output_b,
        size_t num_elements,
        uint32_t* h_histogram)    // [ULP_HIST_BINS] — caller-allocated
    {
        cudaSetDevice(sm86_devices_.empty() ? 0 : sm86_devices_[0]);

        uint64_t* d_ulp = nullptr;
        ULPStats* d_stats = nullptr;
        uint32_t* d_hist = nullptr;

        cudaMalloc(&d_ulp,   num_elements * sizeof(uint64_t));
        cudaMalloc(&d_stats, sizeof(ULPStats));
        cudaMalloc(&d_hist,  ULP_HIST_BINS * sizeof(uint32_t));
        cudaMemset(d_stats, 0, sizeof(ULPStats));
        cudaMemset(d_hist,  0, ULP_HIST_BINS * sizeof(uint32_t));

        const int grid = static_cast<int>((num_elements + 255) / 256);

        ULPDistanceKernel<256><<<grid, 256>>>(
            d_output_a, d_output_b, d_ulp, d_stats,
            static_cast<uint32_t>(num_elements));

        ULPHistogramKernel<256><<<grid, 256>>>(
            d_ulp, d_hist, static_cast<uint32_t>(num_elements));

        cudaMemcpy(h_histogram, d_hist, ULP_HIST_BINS * sizeof(uint32_t),
                   cudaMemcpyDeviceToHost);

        ULPStats stats;
        cudaMemcpy(&stats, d_stats, sizeof(ULPStats), cudaMemcpyDeviceToHost);

        printf("[Crucible] ULP analysis: max=%llu nonzero=%u/%u mean=%.1f\n",
               (unsigned long long)stats.max_ulp, stats.count_nonzero,
               static_cast<uint32_t>(num_elements),
               stats.sum_ulp / fmaxf(1.0f, static_cast<float>(num_elements)));

        cudaFree(d_ulp);
        cudaFree(d_stats);
        cudaFree(d_hist);
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
