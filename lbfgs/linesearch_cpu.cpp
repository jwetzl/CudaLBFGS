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
 * File linesearch.cpp: Implementation of cpu_linesearch.
 *
 **/

#ifdef LBFGS_BUILD_CPU_IMPLEMENTATION

#include "linesearch_cpu.h"

#include <cmath>

bool cpu_linesearch(VectorX &xk, VectorX &z,
					cpu_cost_function *cpucf, floatdouble &fk, VectorX &gk,
					size_t &evals, const VectorX &gkm1, const floatdouble &fkm1,
					lbfgs::status &stat, floatdouble &step, size_t maxEvals)
{
	const floatdouble c1 = 1e-4f;
	const floatdouble c2 = 0.9f;

	const floatdouble alpha_0     = 0.0;
	const floatdouble phi_0       = fk;
	const floatdouble phi_prime_0 = z.dot(gk);

	if (phi_prime_0 >= 0.0)
	{
		stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
		return false;
	}

	const floatdouble alpha_max = 1e8;

	floatdouble alpha     = 1.0;
	floatdouble alpha_old = 0.0;
	bool second_iter = false;

	floatdouble alpha_low, alpha_high;
	floatdouble phi_low, phi_high;
	floatdouble phi_prime_low, phi_prime_high;

	for (;;)
	{
		xk += (alpha - alpha_old) * z;

		cpucf->cpu_f_gradf(xk.data(), &fk, gk.data());

		++evals;

		const floatdouble phi_alpha = fk;
		const floatdouble phi_prime_alpha = z.dot(gk);

		const bool armijo_violated = (phi_alpha > phi_0 + c1 * alpha * phi_prime_0 || (second_iter && phi_alpha >= phi_0));
		const bool strong_wolfe    = (std::abs(phi_prime_alpha) <= -c2 * phi_prime_0);

		// If both Armijo and Strong Wolfe hold, we're done
		if (!armijo_violated && strong_wolfe)
		{
			step = alpha;
			return true;
		}

		if (evals >= maxEvals)
		{
			stat = lbfgs::LBFGS_REACHED_MAX_EVALS;
			return false;
		}

		// If Armijio condition is violated, we've bracketed a viable minimum
		// Interval is [alpha_0, alpha]
		if (armijo_violated)
		{
			alpha_low      = alpha_0;
			alpha_high     = alpha;
			phi_low        = phi_0;
			phi_high       = phi_alpha;
			phi_prime_low  = phi_prime_0;
			phi_prime_high = phi_prime_alpha;
			break;
		}

		// If the directional derivative at alpha is positive, we've bracketed a viable minimum
		// Interval is [alpha, alpha_0]
		if (phi_prime_alpha >= 0)
		{
			alpha_low      = alpha;
			alpha_high     = alpha_0;
			phi_low        = phi_alpha;
			phi_high       = phi_0;
			phi_prime_low  = phi_prime_alpha;
			phi_prime_high = phi_prime_0;
			break;
		}

		// Else look to the "right" of alpha for a viable minimum
		floatdouble alpha_new = alpha + 4 * (alpha - alpha_old);
		alpha_old = alpha;
		alpha     = alpha_new;

		if (alpha > alpha_max)
		{
			stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
			return false;
		}

		second_iter = true;
	}

	// The minimum is now bracketed in [alpha_low, alpha_high]
	// Find it...
	size_t tries = 0;
	const size_t minTries = 10;

	while (true)
	{
		tries++;
		alpha_old = alpha;

		// Quadratic interpolation:
		// Least-squares fit a parabola to (alpha_low, phi_low),
		// (alpha_high, phi_high) with gradients phi_prime_low and
		// phi_prime_high and select the minimum of that parabola as
		// the new alpha

		alpha  = 0.5f * (alpha_low + alpha_high);
		alpha += (phi_high - phi_low) / (phi_prime_low - phi_prime_high);

		if (alpha < alpha_low || alpha > alpha_high)
			alpha = 0.5f * (alpha_low + alpha_high);

		xk += (alpha - alpha_old) * z;

		cpucf->cpu_f_gradf(xk.data(), &fk, gk.data());

		++evals;

		const floatdouble phi_j = fk;
		const floatdouble phi_prime_j = z.dot(gk);

		const bool armijo_violated = (phi_j > phi_0 + c1 * alpha * phi_prime_0 || phi_j >= phi_low);
		const bool strong_wolfe = (std::abs(phi_prime_j) <= -c2 * phi_prime_0);

		if (!armijo_violated && strong_wolfe)
		{
			// The Armijo and Strong Wolfe conditions hold
			step = alpha;
			break;
		}
		else if (std::abs(alpha_high - alpha_low) < 1e-5 && tries > minTries)
		{
			// The search interval has become too small
			stat = lbfgs::LBFGS_LINE_SEARCH_FAILED;
			return false;
		}
		else if (armijo_violated)
		{
			alpha_high     = alpha;
			phi_high       = phi_j;
			phi_prime_high = phi_prime_j;
		}
		else
		{
			if (phi_prime_j * (alpha_high - alpha_low) >= 0)
			{
				alpha_high     = alpha_low;
				phi_high       = phi_low;
				phi_prime_high = phi_prime_low;
			}

			alpha_low     = alpha;
			phi_low       = phi_j;
			phi_prime_low = phi_prime_j;
		}

		if (evals >= maxEvals)
		{
			stat = lbfgs::LBFGS_REACHED_MAX_EVALS;
			return false;
		}
	}

	return true;
}

#endif

