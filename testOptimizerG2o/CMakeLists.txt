cmake_minimum_required(VERSION 3.5)

project(testOptimizerG2o LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)
# Eigen3
find_package( Eigen3 REQUIRED )
# 寻找G2O
find_package( G2O REQUIRED )
#Sophus
find_package( Sophus REQUIRED)
#Cholmod
find_package( Cholmod REQUIRED)

INCLUDE_DIRECTORIES(
                ${CHOLMOD_INCLUDE_DIR}
                ${OpenCV_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIRS}
                ${Sophus_INCLUDE_DIRS})

add_executable(testOptimizerG2o main.cpp)
TARGET_LINK_LIBRARIES(testOptimizerG2o
         ${OpenCV_LIBS}
         ${Sophus_LIBRARIES}
         ${CHOLMOD_LIBRARIES}
         g2o_core g2o_csparse_extension g2o_types_sim3 g2o_stuff g2o_types_sba
         )

 message(STATUS "OPENCV is :${OpenCV_INCLUDE_DIRS}")
 message(STATUS "OPENCV version is :${OpenCV_VERSION}")
 message(STATUS "the cholmod is : ${CHOLMOD_INCLUDE_DIR} ${CHOLMOD_LIBRARIES}")
