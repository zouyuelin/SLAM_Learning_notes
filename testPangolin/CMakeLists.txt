cmake_minimum_required(VERSION 3.5)

project(testPangolin LANGUAGES CXX)

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# Eigen3
find_package( Eigen3 REQUIRED )
# pangoling
find_package( Pangolin REQUIRED)
#Sophus
find_package( Sophus REQUIRED)
# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
INCLUDE_DIRECTORIES(
                ${Pangolin_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIRS}
                ${Sophus_INCLUDE_DIRS}
                ${OpenCV_INCLUDE_DIRS})

add_executable(testPangolin main.cpp)

TARGET_LINK_LIBRARIES(testPangolin
         ${Pangolin_LIBRARIES}
         ${Boost_LIBRARIES}
         ${Sophus_LIBRARIES}
         ${OpenCV_LIBS})
