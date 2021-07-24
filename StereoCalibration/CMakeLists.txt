cmake_minimum_required(VERSION 2.8)

project(StereoCalibration)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_COMPILER "clang++")

find_package(OpenCV  REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

add_executable(openCamera "openCamera.cpp")
target_link_libraries(openCamera ${OpenCV_LIBS})

add_executable(calibrate "calibrate.cpp")
target_link_libraries(calibrate ${OpenCV_LIBS})

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv lib: ${OpenCV_LIBS}")
message(STATUS "opencv include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dir: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv config path : ${OpenCV_CONFIG_PATH}")
message(STATUS "opencv  path : ${OpenCV_DIR}")
