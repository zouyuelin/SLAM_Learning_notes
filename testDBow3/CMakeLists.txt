cmake_minimum_required(VERSION 3.5)

project(testDBow3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package( OpenCV 4.3.0 REQUIRED)
# DBow3
set( DBoW3_INCLUDE_DIRS "/usr/local/include/" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.a" )

INCLUDE_DIRECTORIES(
                ${OpenCV_INCLUDE_DIRS}
                ${DBoW3_INCLUDE_DIRS})

add_executable(creatDBow3vocabulary training_vocabulary.cpp)
TARGET_LINK_LIBRARIES(creatDBow3vocabulary
         ${OpenCV_LIBS}
         ${DBoW3_LIBS})

add_executable(compare_score compare_score.cpp)
TARGET_LINK_LIBRARIES(compare_score
         ${OpenCV_LIBS}
         ${DBoW3_LIBS})

