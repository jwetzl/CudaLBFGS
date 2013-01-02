cuda_add_executable(rosenbrock rosenbrock.cu)
target_link_libraries(rosenbrock lbfgs)

cuda_add_executable(rosenbrock_nd rosenbrock_nd.cu)
target_link_libraries(rosenbrock_nd lbfgs)
