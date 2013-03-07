====================================================
   ___ _   _ ___   _     _       ___ ___ ___ ___ 
  / __| | | |   \ /_\   | |  ___| _ ) __/ __/ __|
 | (__| |_| | |) / _ \  | |_|___| _ \ _| (_ \__ \
  \___|\___/|___/_/ \_\ |____|  |___/_| \___|___/
                                                2012
     by Jens Wetzl           (jens.wetzl@fau.de)
    and Oliver Taubmann (oliver.taubmann@fau.de)

This work is licensed under a Creative Commons
Attribution 3.0 Unported License. (CC-BY)
http://creativecommons.org/licenses/by/3.0/
====================================================

The CUDA L-BFGS library offers GPU based nonlinear
minimization implementing the L-BFGS method in CUDA.

There is no publication available that covers this 
library exclusively, but you may consider citing the 
paper it was introduced in:

Wetzl, J., Taubmann, O., Haase, S., Köhler, T., 
Kraus, M., and Hornegger, J. (2013). GPU-Accelerated 
Time-of-Flight Super-Resolution for Image-Guided 
Surgery. In Meinzer, H.-P., Deserno, T. M., Handels, 
H., and Tolxdorff, T., editors, Bildverarbeitung für 
die Medizin 2013, Informatik aktuell, pages 21–26. 
Springer Berlin Heidelberg.

====================================================
  BUILDING
====================================================

To build (and, if desired, install) the library,
you will need CMake (http://cmake.org). The default
settings should be fine for regular use, but there
are lots of options, e.g. you can

- build a reference implementetation on CPU with
  either float or double precision (requires Eigen),

- build test cases,

- enable error checking, verbose output and timing

- build example projects that demonstrate how the
  library is used (cf. /projects directory).

====================================================
  INCLUDING THE LIBRARY IN YOUR PROJECTS
====================================================

If you use CMake for your project, including the
CudaLBFGS library is jaw-droppingly easy. In your 
CMakeLists.txt file, add:

  find_package(CudaLBFGS REQUIRED)
  include_directories(${CUDALBDFS_INCLUDE_DIRS})
  # ...
  target_link_libraries(YourExecutable
                        ${CUDALBFGS_LIBRARIES})

If you installed the CudaLBFGS library in a non-
standard location, you may also have to set 
either the environment variable CMAKE_PREFIX_PATH
or the CMake variable CUDALBFGS_DIR.

====================================================
  USAGE
====================================================

The basic approach can be described as follows:

1. Implement your cost function in a class that
   inherits from the appropiate base class
   declared in cost_function.h

2. Create an object of class lbfgs (lbfgs.h)
   passing an object of your cost function class
   in the constructor. Adjust settings of lbfgs
   to your liking.
   
3. Run minimization providing an initial guess
   for the solution. Check the return code
   to know which stopping criterion was fulfilled.
