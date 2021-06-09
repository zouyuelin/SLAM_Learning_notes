cmake_minimum_required( VERSION 2.8 )
project( g2o_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )

#PCL
find_package(PCL REQUIRED)
add_definitions(${PCL_DEFINITIONS})   
#Qt5
find_package(Qt5 COMPONENTS Widgets Core Gui Network REQUIRED)
# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)
# Eigen3
find_package( Eigen3 REQUIRED )
# 寻找G2O
find_package( G2O REQUIRED )
# pangoling
find_package( Pangolin REQUIRED)
#Sophus
find_package( Sophus REQUIRED)
#boost
find_package( Boost REQUIRED)

INCLUDE_DIRECTORIES(
                ${PROJECT_SOURCE_DIR}
                ${Pangolin_INCLUDE_DIRS}
                ${OpenCV_INCLUDE_DIRS}
                ${PCL_INCLUDE_DIRS}
                ${BOOST_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIRS}
                ${Sophus_INCLUDE_DIRS})


add_executable(${PROJECT_NAME}  g2o_fitting.cpp)
add_executable(curve_lambda curve_lambda.cpp)

TARGET_LINK_LIBRARIES(${PROJECT_NAME} 
				      ${VTK_LIBRARIES} 
				      ${OpenCV_LIBS} 
				      ${Pangolin_LIBRARIES}
				      ${Boost_LIBRARIES}
				      ${Sophus_LIBRARIES}
				      ${PCL_LIBRARIES}
				      Qt5::Core Qt5::Widgets Qt5::Gui Qt5::Network
                                      g2o_core g2o_csparse_extension g2o_types_sim3 g2o_stuff g2o_types_sba
				      )
					  
message(STATUS "OPENCV is :${OpenCV_INCLUDE_DIRS}")
message(STATUS "OPENCV version is :${OpenCV_VERSION}")


