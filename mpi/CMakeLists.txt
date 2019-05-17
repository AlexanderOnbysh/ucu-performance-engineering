cmake_minimum_required(VERSION 3.13)
project(mpi)

include_directories({MPI_INCLUDE_PATH})

set(CMAKE_CXX_STANDARD 14)

add_executable(mpi main.cpp)