cmake_minimum_required(VERSION 3.5)

project(CameraTransformer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigen3
find_package( Eigen3 REQUIRED )

include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(CameraTransformer
    main.cpp
    Transformer.cpp
    Transformer.h)
