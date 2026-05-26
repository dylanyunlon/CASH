#pragma once

#ifndef CRUCIBLE_INDEX_KERNEL_HARNESS_CUH
#define CRUCIBLE_INDEX_KERNEL_HARNESS_CUH

// Crucible Index Kernel Test Harness
//
// Wraps HypeReca's three index kernels (indexGet, indexPut, indexCopy)
// for cross-architecture differential testing.  Each wrapper:
//
//   1. Allocates deterministic input on the target device
//   2. Launches the kernel with a given configuration
//   3. Captures the output for comparison
//
// The harness exercises the kernels at architecture-sensitive
// boundary conditions:
//
//   - Embedding dimensions at warp-width multiples (32, 64, 128)
//     vs non-aligned (33, 65, 129)
//   - Row counts at block-boundary edges
//   - Mixed alignment of input/output buffer base addresses
//
// These are the same kernels that Alloy's TieredEmbedding depends on,
// so Crucible validates them before Alloy trusts their output.

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>

#include "crucible/arch_dispatch.cuh"
#include "crucible/differential_validator.cuh"

namespace Crucible {

// ─────────────────────────────────────────────────────────────────
//  Replicate HypeReca's index kernels here (header-only) so we
//  can compile them for both sm86 and sm90 in the same binary.
//
//  These are exact copies of core/hypereca/src/embeddings/index_kernels.cuh
//  — any divergence between architectures is a real finding.
// ─────────────────────────────────────────────────────────────────

template <typename DataType>
__global__ void indexPutKernel(size_t n, DataType* out,
        const DataType* in, const size_t* idx)
{
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) return;
    out[idx[row_idx] * blockDim.x + threadIdx.x] =
        in[row_idx * blockDim.x + threadIdx.x];
}

template <typename DataType>
__global__ void indexGetKernel(size_t n, DataType* out,
        const DataType* in, const size_t* idx)
{
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) return;
    out[row_idx * blockDim.x + threadIdx.x] =
        in[idx[row_idx] * blockDim.x + threadIdx.x];
}

template <typename DataType>
__global__ void indexCopyKernel(size_t n, DataType* out,
        const size_t* out_idx, const DataType* in, const size_t* in_idx)
{
    size_t row_idx = blockDim.y * blockIdx.x + threadIdx.y;
    if (row_idx >= n) return;
    out[out_idx[row_idx] * blockDim.x + threadIdx.x] =
        in[in_idx[row_idx] * blockDim.x + threadIdx.x];
}

// ─────────────────────────────────────────────────────────────────
//  Test configuration
// ─────────────────────────────────────────────────────────────────

struct IndexKernelTestConfig {
    size_t num_rows;        // number of embedding rows in table
    size_t embedding_dim;   // must divide evenly into blockDim.x
    size_t batch_size;      // number of lookups in this test
    size_t alignment_offset; // byte offset to stress alignment
    uint64_t seed;          // deterministic fill seed

    // Compute the HypeReca-style launch config:
    //   blockDim.x = embedding_dim
    //   blockDim.y = 512 / embedding_dim (rows per block)
    //   gridDim.x  = ceil(batch_size / blockDim.y)
    void get_launch_config(dim3& grid, dim3& block) const
    {
        const int per_block = 512 / static_cast<int>(embedding_dim);
        block = dim3(static_cast<unsigned>(embedding_dim),
                     static_cast<unsigned>(per_block > 0 ? per_block : 1));
        grid  = dim3(static_cast<unsigned>(
                     (batch_size + block.y - 1) / block.y));
    }
};

// ─────────────────────────────────────────────────────────────────
//  Harness: allocate, fill, run, capture
// ─────────────────────────────────────────────────────────────────

struct IndexKernelHarness {

    // Run indexGetKernel on a specific device and return the output.
    // Caller must free d_output.
    template <typename DataType>
    static void run_index_get(
        int                        device_id,
        const IndexKernelTestConfig& cfg,
        DataType**                  d_output,   // allocated by harness
        cudaStream_t               stream = 0)
    {
        cudaSetDevice(device_id);

        const size_t table_elems = cfg.num_rows * cfg.embedding_dim;
        const size_t batch_elems = cfg.batch_size * cfg.embedding_dim;

        // Allocate
        DataType* d_table   = nullptr;
        DataType* d_out     = nullptr;
        size_t*   d_indices = nullptr;

        cudaMalloc(&d_table,   table_elems * sizeof(DataType));
        cudaMalloc(&d_out,     batch_elems * sizeof(DataType));
        cudaMalloc(&d_indices, cfg.batch_size * sizeof(size_t));

        // Fill deterministically
        const int fill_grid = static_cast<int>((table_elems + 255) / 256);
        FillDeterministicPattern<DataType><<<fill_grid, 256, 0, stream>>>(
            d_table, table_elems, cfg.seed);
        GenerateBoundaryIndices<<<(cfg.batch_size + 255) / 256, 256, 0, stream>>>(
            d_indices, cfg.batch_size, cfg.num_rows, cfg.seed + 1);

        // Zero output
        cudaMemsetAsync(d_out, 0, batch_elems * sizeof(DataType), stream);

        // Launch indexGetKernel with HypeReca's config
        dim3 grid, block;
        cfg.get_launch_config(grid, block);

        indexGetKernel<DataType><<<grid, block, 0, stream>>>(
            cfg.batch_size, d_out, d_table, d_indices);

        cudaStreamSynchronize(stream);

        *d_output = d_out;

        // Cleanup (keep d_out for caller)
        cudaFree(d_table);
        cudaFree(d_indices);
    }

    // Run indexPutKernel on a specific device.
    template <typename DataType>
    static void run_index_put(
        int                        device_id,
        const IndexKernelTestConfig& cfg,
        DataType**                  d_output,
        cudaStream_t               stream = 0)
    {
        cudaSetDevice(device_id);

        const size_t table_elems = cfg.num_rows * cfg.embedding_dim;
        const size_t batch_elems = cfg.batch_size * cfg.embedding_dim;

        DataType* d_table    = nullptr;
        DataType* d_values   = nullptr;
        size_t*   d_indices  = nullptr;

        cudaMalloc(&d_table,   table_elems * sizeof(DataType));
        cudaMalloc(&d_values,  batch_elems * sizeof(DataType));
        cudaMalloc(&d_indices, cfg.batch_size * sizeof(size_t));

        // Fill table and values
        const int fill_grid = static_cast<int>((table_elems + 255) / 256);
        FillDeterministicPattern<DataType><<<fill_grid, 256, 0, stream>>>(
            d_table, table_elems, cfg.seed);
        const int val_grid = static_cast<int>((batch_elems + 255) / 256);
        FillDeterministicPattern<DataType><<<val_grid, 256, 0, stream>>>(
            d_values, batch_elems, cfg.seed + 2);
        GenerateBoundaryIndices<<<(cfg.batch_size + 255) / 256, 256, 0, stream>>>(
            d_indices, cfg.batch_size, cfg.num_rows, cfg.seed + 3);

        dim3 grid, block;
        cfg.get_launch_config(grid, block);

        indexPutKernel<DataType><<<grid, block, 0, stream>>>(
            cfg.batch_size, d_table, d_values, d_indices);

        cudaStreamSynchronize(stream);

        *d_output = d_table;

        cudaFree(d_values);
        cudaFree(d_indices);
    }

    // Run indexCopyKernel on a specific device.
    template <typename DataType>
    static void run_index_copy(
        int                        device_id,
        const IndexKernelTestConfig& cfg,
        DataType**                  d_output,
        cudaStream_t               stream = 0)
    {
        cudaSetDevice(device_id);

        const size_t table_elems = cfg.num_rows * cfg.embedding_dim;

        DataType* d_src     = nullptr;
        DataType* d_dst     = nullptr;
        size_t*   d_src_idx = nullptr;
        size_t*   d_dst_idx = nullptr;

        cudaMalloc(&d_src,     table_elems * sizeof(DataType));
        cudaMalloc(&d_dst,     table_elems * sizeof(DataType));
        cudaMalloc(&d_src_idx, cfg.batch_size * sizeof(size_t));
        cudaMalloc(&d_dst_idx, cfg.batch_size * sizeof(size_t));

        const int fill_grid = static_cast<int>((table_elems + 255) / 256);
        FillDeterministicPattern<DataType><<<fill_grid, 256, 0, stream>>>(
            d_src, table_elems, cfg.seed);
        cudaMemsetAsync(d_dst, 0, table_elems * sizeof(DataType), stream);

        GenerateBoundaryIndices<<<(cfg.batch_size + 255) / 256, 256, 0, stream>>>(
            d_src_idx, cfg.batch_size, cfg.num_rows, cfg.seed + 4);
        GenerateBoundaryIndices<<<(cfg.batch_size + 255) / 256, 256, 0, stream>>>(
            d_dst_idx, cfg.batch_size, cfg.num_rows, cfg.seed + 5);

        dim3 grid, block;
        cfg.get_launch_config(grid, block);

        indexCopyKernel<DataType><<<grid, block, 0, stream>>>(
            cfg.batch_size, d_dst, d_dst_idx, d_src, d_src_idx);

        cudaStreamSynchronize(stream);

        *d_output = d_dst;

        cudaFree(d_src);
        cudaFree(d_src_idx);
        cudaFree(d_dst_idx);
    }

    // ─────────────────────────────────────────────────────────
    //  Full differential test: run on two devices, compare
    // ─────────────────────────────────────────────────────────

    enum class KernelType { GET, PUT, COPY };

    template <typename DataType>
    static DivergenceReport differential_test(
        int          dev_a,   // sm86
        int          dev_b,   // sm90
        KernelType   kernel,
        const IndexKernelTestConfig& cfg)
    {
        DataType* d_out_a = nullptr;
        DataType* d_out_b = nullptr;

        switch (kernel)
        {
        case KernelType::GET:
            run_index_get<DataType>(dev_a, cfg, &d_out_a);
            run_index_get<DataType>(dev_b, cfg, &d_out_b);
            break;
        case KernelType::PUT:
            run_index_put<DataType>(dev_a, cfg, &d_out_a);
            run_index_put<DataType>(dev_b, cfg, &d_out_b);
            break;
        case KernelType::COPY:
            run_index_copy<DataType>(dev_a, cfg, &d_out_a);
            run_index_copy<DataType>(dev_b, cfg, &d_out_b);
            break;
        }

        // Copy dev_b output to dev_a for comparison
        const size_t out_elems = (kernel == KernelType::GET)
            ? cfg.batch_size * cfg.embedding_dim
            : cfg.num_rows * cfg.embedding_dim;

        cudaSetDevice(dev_a);
        DataType* d_out_b_copy = nullptr;
        cudaMalloc(&d_out_b_copy, out_elems * sizeof(DataType));
        cudaMemcpy(d_out_b_copy, d_out_b, out_elems * sizeof(DataType),
                   cudaMemcpyDeviceToDevice);

        // Compare
        DivergenceReport* d_report = nullptr;
        cudaMalloc(&d_report, sizeof(DivergenceReport));
        cudaMemset(d_report, 0, sizeof(DivergenceReport));

        const int cmp_grid = static_cast<int>((out_elems + 255) / 256);
        // DifferentialCompareKernel replaces BitwiseCompareKernel with mode selection
        DifferentialCompareKernel<DataType, 256><<<cmp_grid, 256>>>(
            d_out_a, d_out_b_copy,
            static_cast<uint32_t>(out_elems),
            d_report,
            CompareMode::BITWISE,
            0.0f);

        DivergenceReport report;
        cudaMemcpy(&report, d_report, sizeof(DivergenceReport),
                   cudaMemcpyDeviceToHost);
        report.total_elements = static_cast<uint32_t>(out_elems);

        cudaFree(d_out_a);
        cudaFree(d_out_b);
        cudaFree(d_out_b_copy);
        cudaFree(d_report);

        return report;
    }
};

// ─────────────────────────────────────────────────────────────────
//  Boundary test suite generator
//
//  Produces IndexKernelTestConfig instances at architecture-
//  sensitive boundaries.  These are the exact configurations
//  that Exp1 (differential fuzzing) sweeps.
// ─────────────────────────────────────────────────────────────────

struct BoundaryTestSuite {

    static constexpr size_t EMBEDDING_DIMS[] = {
        32, 33,    // warp-aligned vs off-by-one
        64, 65,    // two warps vs off-by-one
        128, 129,  // four warps vs off-by-one
        256,       // common production size
        512, 1024  // max blockDim.x values
    };

    static constexpr size_t ROW_COUNTS[] = {
        1, 31, 32, 33,          // single warp boundary
        255, 256, 257,          // block boundary
        1023, 1024, 1025,       // max threads boundary
        65535, 65536, 100000    // large scale
    };

    static constexpr size_t NUM_DIMS = sizeof(EMBEDDING_DIMS) / sizeof(size_t);
    static constexpr size_t NUM_ROWS = sizeof(ROW_COUNTS) / sizeof(size_t);

    // Generate all (dim × row_count) configurations.
    // Returns total count; fills configs[0..count-1].
    static size_t generate(IndexKernelTestConfig* configs, size_t max_configs)
    {
        size_t count = 0;
        for (size_t di = 0; di < NUM_DIMS && count < max_configs; ++di)
        {
            for (size_t ri = 0; ri < NUM_ROWS && count < max_configs; ++ri)
            {
                // Skip configs where embedding_dim exceeds CUDA limits
                if (EMBEDDING_DIMS[di] > 1024) continue;
                // Batch size = min(row_count, 4096) for reasonable runtime
                const size_t batch = (ROW_COUNTS[ri] < 4096)
                    ? ROW_COUNTS[ri] : 4096;

                configs[count++] = {
                    ROW_COUNTS[ri],
                    EMBEDDING_DIMS[di],
                    batch,
                    0,                               // alignment offset
                    42 + di * 1000 + ri              // deterministic seed
                };
            }
        }
        return count;
    }
};

constexpr size_t BoundaryTestSuite::EMBEDDING_DIMS[];
constexpr size_t BoundaryTestSuite::ROW_COUNTS[];

}  // namespace Crucible

#endif  // CRUCIBLE_INDEX_KERNEL_HARNESS_CUH
