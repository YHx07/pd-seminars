cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_COMPILER mpic++)
set(CMAKE_C_COMPILER mpicc)

project(MpiIsendRecv)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MpiIsendRecv_SOURCE_DIR}/bin)

add_executable(MpiIsendRecv main.cpp)
