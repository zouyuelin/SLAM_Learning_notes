cmake_minimum_required(VERSION 3.5)

project(loadYamlAndUndisort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER "clang++")

# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)
# Eigen3
find_package( Eigen3 REQUIRED )


INCLUDE_DIRECTORIES(
                ${OpenCV_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIRS})

add_executable(loadYamlAndUndisort main.cpp)

TARGET_LINK_LIBRARIES(loadYamlAndUndisort
         ${OpenCV_LIBS})
