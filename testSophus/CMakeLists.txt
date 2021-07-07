cmake_minimum_required(VERSION 3.5)

project(testSophus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set( CMAKE_BUILD_TYPE "Release" )

#Sophus
find_package( Sophus REQUIRED)
# Eigen3
find_package( Eigen3 REQUIRED )

INCLUDE_DIRECTORIES(${Sophus_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})

add_executable(testSophus main.cpp)

TARGET_LINK_LIBRARIES(testSophus ${Sophus_LIBRARIES})
