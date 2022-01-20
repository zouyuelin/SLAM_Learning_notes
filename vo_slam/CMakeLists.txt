cmake_minimum_required(VERSION 3.5)

project(VO_Slam LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(CMAKE_CXX_COMPILER "clang++")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
#******openMp 多线程****
find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -pthread -O3 -Wall") # ${OpenMP_CXX_FLAGS} = -fopenmp
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
#boost
find_package( Boost REQUIRED COMPONENTS system filesystem)
# OpenCV
find_package( OpenCV 4.5.1 EXACT REQUIRED)
# Eigen3
find_package( Eigen3 3.3.7 EXACT REQUIRED )
# 寻找G2O
find_package( G2O REQUIRED )
#Sophus
find_package( Sophus REQUIRED)
#Cholmod
find_package( Cholmod REQUIRED)
# pangoling
find_package( Pangolin REQUIRED)

INCLUDE_DIRECTORIES(
                ${CHOLMOD_INCLUDE_DIR}
                ${OpenCV_INCLUDE_DIRS}
                ${Pangolin_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIR}
                ${Sophus_INCLUDE_DIRS}
                ${BOOST_INCLUDE_DIR}
                ${ONNXRUNTIME_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} main.cpp vo_slam.hpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
         ${OpenCV_LIBS}
         ${Sophus_LIBRARIES}
         ${CHOLMOD_LIBRARIES}
         g2o_core g2o_types_slam3d g2o_solver_csparse g2o_stuff g2o_csparse_extension g2o_types_sim3 g2o_types_sba
         ${Pangolin_LIBRARIES}
         ${Boost_LIBRARIES}
         )

 message(STATUS "OPENCV is :${OpenCV_INCLUDE_DIRS}")
 message(STATUS "OPENCV version is :${OpenCV_VERSION}")
 message(STATUS "the cholmod is ${CHOLMOD_INCLUDE_DIR} ${CHOLMOD_LIBRARIES} ")
 message(STATUS "Eigen and boost :${EIGEN3_INCLUDE_DIR} ${Boost_INCLUDE_DIRS} ")
