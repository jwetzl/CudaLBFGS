cmake_minimum_required(VERSION 2.8)
project(CudaLBFGS)

set(CUDALBFGS_MAJOR_VERSION 1)
set(CUDALBFGS_MINOR_VERSION 0)
set(CUDALBFGS_PATCH_VERSION 2)
set(CUDALBFGS_VERSION "${CUDALBFGS_MAJOR_VERSION}.${CUDALBFGS_MINOR_VERSION}.${CUDALBFGS_PATCH_VERSION}")

enable_testing()

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE CudaLBFGS)

# Put all executables and libraries into the top level
# binary directory instead of into subfolders
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE})

# Where to look for Find*.cmake
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

find_package(CUDA REQUIRED)

include_directories(include)
include_directories(include/CudaLBFGS)

include(CheckComputeCapability.cmake)

# ---

option(LBFGS_CPU_DOUBLE_PRECISION "Enable double precision mode for CPU implementation")

if (LBFGS_CPU_DOUBLE_PRECISION)
        add_definitions(-DLBFGS_CPU_DOUBLE_PRECISION)
endif()

# ---

file(GLOB HEADERS include/CudaLBFGS/*.h)

# ---

add_subdirectory(lbfgs)

# ---

option(BUILD_PROJECTS "Build example projects")

if (BUILD_PROJECTS)
        add_subdirectory(projects)
endif()
