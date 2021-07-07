cmake_minimum_required(VERSION 3.5)

project(testEigen LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3
find_package( Eigen3 REQUIRED )

INCLUDE_DIRECTORIES(
    ${EIGEN3_INCLUDE_DIRS})

add_executable(testEigen main.cpp)
