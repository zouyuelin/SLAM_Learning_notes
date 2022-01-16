cmake_minimum_required(VERSION 2.8)
project(ONNXobjDetect)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_COMPILER clang++)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_INCLUDE_CURRENT_DIR ON)


#find the opencv and the qt5
find_package(OpenCV 4.5.1  REQUIRED)
find_package(Qt5 COMPONENTS Widgets Core Gui Network REQUIRED)

#onnxruntime
set(ONNXRUNTIME_ROOT_PATH /home/zyl/ubuntu/tensorRT/onnxruntime-master)
set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/onnxruntime
                             ${ONNXRUNTIME_ROOT_PATH}/include/onnxruntime/core/session/)
set(ONNXRUNTIME_LIB ${ONNXRUNTIME_ROOT_PATH}/build/Linux/Release/libonnxruntime.so)
include_directories(${ONNXRUNTIME_INCLUDE_DIRS})


add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    Qt5::Core
    Qt5::Widgets
    Qt5::Gui
    Qt5::Network ${ONNXRUNTIME_LIB})
message(STATUS "the ONN is ${ONNXRUNTIME_INCLUDE_DIRS}")
