/**
 *   ___ _   _ ___   _     _       ___ ___ ___ ___
 *  / __| | | |   \ /_\   | |  ___| _ ) __/ __/ __|
 * | (__| |_| | |) / _ \  | |_|___| _ \ _| (_ \__ \
 *  \___|\___/|___/_/ \_\ |____|  |___/_| \___|___/
 *                                               2012
 *     by Jens Wetzl           (jens.wetzl@fau.de)
 *    and Oliver Taubmann (oliver.taubmann@fau.de)
 *
 * This work is licensed under a Creative Commons
 * Attribution 3.0 Unported License. (CC-BY)
 * http://creativecommons.org/licenses/by/3.0/
 *
 * File linesearch.h: Line search for CPU implementation.
 *
 **/

#ifndef LINESEARCH_H
#define LINESEARCH_H

#ifdef LBFGS_BUILD_CPU_IMPLEMENTATION

#include "cost_function.h"
#include "lbfgs.h"

#include <iostream>
using namespace std;

#include <Eigen/Dense>
using namespace Eigen;

#ifdef LBFGS_CPU_DOUBLE_PRECISION
	typedef VectorXd VectorX;
#else
	typedef VectorXf VectorX;
#endif

bool cpu_linesearch(VectorX &xk, VectorX &z, cpu_cost_function *cpucf,
                    floatdouble &fk, VectorX &gk, size_t &evals,
                    const VectorX &gkm1, const floatdouble &fkm1,
                    lbfgs::status &stat, floatdouble &step, size_t maxEvals);

#endif

#endif // LINESEARCH_H
