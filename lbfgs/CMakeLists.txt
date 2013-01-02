# Ensures library headers on windows use
# dllexport instead of dllimport.

if (WIN32)
	add_definitions(-DLBFGS_BUILD_DLL)
endif()

# ---

option(LBFGS_TIMING "Enable timing for LBFGS.")

if (LBFGS_TIMING)
        add_definitions(-DLBFGS_TIMING)
endif()

# ---

option(LBFGS_VERBOSE "Print verbose debug output for L-BFGS library.")

if (LBFGS_VERBOSE)
        add_definitions(-DLBFGS_VERBOSE)
endif()

# ---

option(LBFGS_BUILD_CPU_IMPLEMENTATION "Build CPU implementation (requires Eigen).")

if (LBFGS_BUILD_CPU_IMPLEMENTATION)
        find_package(Eigen REQUIRED)
        include_directories(${Eigen_INCLUDE_DIRS})
        add_definitions(-DLBFGS_BUILD_CPU_IMPLEMENTATION)
endif()

# ---

option(LBFGS_CUDA_ERROR_CHECK "Enable CUDA error checking.")

if (LBFGS_CUDA_ERROR_CHECK)
        add_definitions(-DLBFGS_CUDA_ERROR_CHECK)
endif()

# ---

file(GLOB CU_FILES *.cu)
file(GLOB H_FILES *.h)
file(GLOB CPP_FILES *.cpp)

cuda_add_library(lbfgs SHARED ${CU_FILES} ${CPP_FILES} ${HEADERS} ${H_FILES})

cuda_add_cublas_to_target(lbfgs)


source_group(Headers FILES ${HEADERS})
source_group("CUDA Sources" FILES ${CU_FILES})
source_group("CPP Sources" FILES ${CPP_FILES})
source_group("Headers" FILES ${H_FILES})

# ---
# Installing

# Add all targets to the build-tree export set
export(TARGETS lbfgs
  FILE "${PROJECT_BINARY_DIR}/CudaLBFGSLibraryDepends.cmake")
  
# Create a CudaLBFGSBuildTreeSettings.cmake file for the use from the build tree
configure_file(CudaLBFGSBuildTreeSettings.cmake.in
  "${PROJECT_BINARY_DIR}/CudaLBFGSBuildTreeSettings.cmake" @ONLY)

# Create the CudaLBFGSConfig.cmake

set(INSTALL_LIB_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR include CACHE PATH
  "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")
 
# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

file(RELATIVE_PATH CONF_REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}"
	"${INSTALL_INCLUDE_DIR}")
configure_file(CudaLBFGSConfig.cmake.in
	"${PROJECT_BINARY_DIR}/CudaLBFGSConfig.cmake" @ONLY)
configure_file(CudaLBFGSConfigVersion.cmake.in 
	"${PROJECT_BINARY_DIR}/CudaLBFGSConfigVersion.cmake" @ONLY)

file (GLOB INSTALL_HEADER_FILES ${CMAKE_SOURCE_DIR}/include/CudaLBFGS/*.h)

install(FILES ${INSTALL_HEADER_FILES}
	DESTINATION ${INSTALL_INCLUDE_DIR}/CudaLBFGS
)

install(TARGETS lbfgs
	EXPORT CudaLBFGSLibraryDepends
	RUNTIME DESTINATION ${INSTALL_BIN_DIR}
	LIBRARY DESTINATION ${INSTALL_LIB_DIR}
	ARCHIVE DESTINATION ${INSTALL_LIB_DIR}
	COMPONENT dev
)

# Install the export set for use with the install-tree
install(EXPORT CudaLBFGSLibraryDepends DESTINATION
  "${INSTALL_CMAKE_DIR}"
  COMPONENT dev)
  
install(FILES ${PROJECT_BINARY_DIR}/CudaLBFGSConfig.cmake 
	DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)
install(FILES ${PROJECT_BINARY_DIR}/CudaLBFGSConfigVersion.cmake 
	DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

# ---

option(UNIT_TESTING "Do unit testing (also enable LBFGS_BUILD_CPU_IMPLEMENTATION!).")

if (UNIT_TESTING AND LBFGS_BUILD_CPU_IMPLEMENTATION)
        enable_testing()
        add_subdirectory(tests)
endif()
