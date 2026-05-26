cmake_minimum_required(VERSION 3.21)
project(autolocalize_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_native MODULE
  src/bindings.cpp
  src/pose_scorer.cpp
  src/refine.cpp
)

target_compile_options(_native PRIVATE -O3 -ffast-math)

install(TARGETS _native DESTINATION autolocalize)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/../maps" DESTINATION autolocalize)
