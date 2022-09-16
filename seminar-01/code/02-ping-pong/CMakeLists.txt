cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(MpiPingPong)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiPingPong_SOURCE_DIR}/bin)

add_executable(MpiPingPong main.cpp)
