cmake_minimum_required(VERSION 2.8)

project(calibrationCamera)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV 4.5.1 REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIR})

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

message(STATUS "opencv version: ${OpenCV_VERSION}")
message(STATUS "opencv lib: ${OpenCV_LIBS}")
message(STATUS "opencv include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dir: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv config path : ${OpenCV_CONFIG_PATH}")
message(STATUS "opencv  path : ${OpenCV_DIR}")
