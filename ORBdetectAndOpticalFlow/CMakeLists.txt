cmake_minimum_required(VERSION 3.5)

project(ORBdetect LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# OpenCV
find_package( OpenCV REQUIRED)

INCLUDE_DIRECTORIES(
                ${OpenCV_INCLUDE_DIRS})

add_executable(ORBdetect main.cpp)
TARGET_LINK_LIBRARIES(ORBdetect
         ${OpenCV_LIBS})

 message(STATUS "OPENCV is :${OpenCV_INCLUDE_DIRS}")
 message(STATUS "OPENCV version is :${OpenCV_VERSION}")
