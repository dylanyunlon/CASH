cmake_minimum_required(VERSION 3.18)
project(CASH LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "86;90")

option(CRUCIBLE_BUILD_TESTS "Build Crucible experiment binaries" ON)

# Core Crucible CUDA library
add_subdirectory(crucible)

if(CRUCIBLE_BUILD_TESTS)
    # TODO: experiment runner binaries
endif()
