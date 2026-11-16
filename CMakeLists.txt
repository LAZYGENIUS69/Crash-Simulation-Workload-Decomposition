cmake_minimum_required(VERSION 3.16)
project(CrashSimulationWorkloadDecomposition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# Output directory for all executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

# 1. FP64 Double Precision Solver (Default)
add_executable(crash_solver src_cpp/main.cpp)
target_link_libraries(crash_solver PRIVATE OpenMP::OpenMP_CXX)

# 2. FP32 Single Precision Solver (Memory Bandwidth Sensitivity)
add_executable(crash_solver_fp32 src_cpp/main.cpp)
target_compile_definitions(crash_solver_fp32 PRIVATE USE_FP32)
target_link_libraries(crash_solver_fp32 PRIVATE OpenMP::OpenMP_CXX)

# 3. STREAM Triad Memory Bandwidth Benchmark
add_executable(stream_bench src_cpp/stream_bench.cpp)
target_link_libraries(stream_bench PRIVATE OpenMP::OpenMP_CXX)

