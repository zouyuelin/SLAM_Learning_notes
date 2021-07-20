set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_COMPILER "clang++")

find_package(Pangolin REQUIRED)
# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)
# Eigen3
find_package( Eigen3 REQUIRED )
#Sophus
find_package( Sophus REQUIRED)


INCLUDE_DIRECTORIES(
                ${OpenCV_INCLUDE_DIRS}
                ${EIGEN3_INCLUDE_DIRS}
                ${Sophus_INCLUDE_DIRS})

add_executable(stereoVision stereoVision.cpp)
target_link_libraries(stereoVision ${OpenCV_LIBS} ${Pangolin_LIBRARIES})