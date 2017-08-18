cmake_minimum_required(VERSION 3.8)
project(cpp)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp kalman_filter.cpp kalman_filter.h tracking.cpp tracking.h measurement_package.h)
add_executable(cpp ${SOURCE_FILES})
link_directories(${PROJECT_SOURCE_DIR}/Eigen)