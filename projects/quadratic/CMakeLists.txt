cuda_add_executable(quadratic quadratic.cu)
cuda_add_cublas_to_target(quadratic)
target_link_libraries(quadratic lbfgs)

