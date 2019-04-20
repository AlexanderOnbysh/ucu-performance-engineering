cmake_minimum_required(VERSION 3.13)
project(project_1)

INCLUDE_DIRECTORIES(/opt/OpenBLAS/include)
set(CMAKE_CXX_STANDARD 17)

add_executable(project_1 main.cpp)