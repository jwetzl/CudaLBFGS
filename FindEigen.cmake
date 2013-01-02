find_package(PkgConfig)
pkg_check_modules(PC_Eigen QUIET eigen3)

find_path(Eigen_INCLUDE_DIR Eigen HINTS ${PC_Eigen_INCLUDEDIR} ${PC_Eigen_INCLUDE_DIRS} /usr/include /usr/local/include /opt/local/include PATH_SUFFIXES eigen3)
set(Eigen_INCLUDE_DIRS ${Eigen_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen DEFAULT_MSG Eigen_INCLUDE_DIR)

mark_as_advanced(Eigen_INCLUDE_DIR)