cmake_minimum_required(VERSION 3.10)
project(zero_minor_refactored VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(zero_minor_refactored
    src/main.cpp
    src/core/Matrix.cpp
    src/services/MinorService.cpp
    src/utils/IO.cpp
)

target_include_directories(zero_minor_refactored PRIVATE ${PROJECT_SOURCE_DIR}/include)
