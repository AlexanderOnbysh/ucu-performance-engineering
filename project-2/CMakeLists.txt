cmake_minimum_required(VERSION 3.13)

set(CMAKE_CUDA_COMPILER /usr/bin/nvcc)
set(CUDA_CUDART_LIBRARY /usr/local/cuda/lib64/libcudart.so)

project(project_2 LANGUAGES C CXX CUDA)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
        CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -O3 -gencode arch=compute_22,code=sm_22
)

# Specify include directories
include_directories(
        kernels
        utility
)

# Specify library paths
link_directories(
        /opt/foobar/lib
        /opt/joestuff/lib
)

add_executable(project_2 main.cu)