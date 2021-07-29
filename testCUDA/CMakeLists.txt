cmake_minimum_required(VERSION 3.5)

project(testCUDA)

if(CUDA_ENABLE)
    enable_language(CUDA)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED)
# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)

SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -use_fast_math )#--default-stream legacy 默认流

INCLUDE_DIRECTORIES(
                ${PROJECT_SOURCE_DIR}
                ${OpenCV_INCLUDE_DIRS}
                ${CUDA_INCLUDE_DIRS})

message(STATUS "${CUDA_INCLUDE_DIRS}")

CUDA_ADD_EXECUTABLE(testCUDA main.cu)

target_link_libraries(testCUDA  ${OpenCV_LIBS})
