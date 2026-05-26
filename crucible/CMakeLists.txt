cmake_minimum_required(VERSION 3.18)
project(Crucible CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# ── Multi-architecture compilation (must target BOTH to do differential testing) ──
set(CMAKE_CUDA_ARCHITECTURES "86;90")

find_package(CUDAToolkit REQUIRED)

# ── Crucible kernel validation and scheduling library ──
add_library(crucible_kernels
    crucible/src/crucible_runtime.cu
)

target_include_directories(crucible_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/crucible/include
)

target_link_libraries(crucible_kernels
    CUDA::cudart
)

target_compile_options(crucible_kernels PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        --expt-relaxed-constexpr
        -Xptxas -v
    >
)
