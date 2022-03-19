cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
PROJECT(msf LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#Cmake need to know where to find things
set(msf_CMAKE_PATH ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${msf_CMAKE_PATH}/cmake)

find_package(Eigen3 REQUIRED)
message(${Eigen3_DIR})

find_package(Pangolin)
include_directories(${Pangolin_INCLUDE_DIRS})

add_subdirectory(src)





