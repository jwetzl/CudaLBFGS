#include "lbfgs.h"
#include "matrices.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

class cpu_quadratic : public cpu_cost_function
{
public:
	// A is expected to point to n*n floats
	// b is expected to point to n floats
	cpu_quadratic(const size_t n, const float *A, const float *b, const float c)
	: cpu_cost_function(n)
	, m_A(A)
	, m_b(b)
	, m_c(c) {}

	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		floatdouble xAx = 0.0f;

		for (size_t i = 0; i < m_numDimensions; ++i)
		{
			for (size_t j = 0; j < m_numDimensions; ++j)
			{
				xAx += m_A[i * m_numDimensions + j] * h_x[i] * h_x[j];
			}
		}

		floatdouble bx = 0.0f;

		for (size_t i = 0; i < m_numDimensions; ++i)
		{
			bx += m_b[i] * h_x[i];
		}

		*h_y = xAx + bx + m_c;
	}

	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		for (size_t i = 0; i < m_numDimensions; ++i)
		{
			h_grad[i] = 0.0f;

			for (size_t j = 0; j < m_numDimensions; ++j)
			{
				h_grad[i] += m_A[i * m_numDimensions + j] * h_x[j];
			}

			h_grad[i] *= 2.0f;
			h_grad[i] += m_b[i];
		}
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}

private:
	const float *m_A;
	const float *m_b;
	const float  m_c;
};

bool test(const size_t size, const float *matrix, const float *linear, const float constant,
	const float *minimum, const float epsilon)
{
	float *xstart = new float[size];
	srand(0xDEADBEEF);

	for (size_t i = 0; i < size; ++i)
	{
		xstart[i] = minimum[i] + 40.0f * float(rand()) / float(RAND_MAX) - 20.0f;
	}

	cpu_quadratic cost(size, matrix, linear, constant);
	lbfgs minimizer(cost);
	minimizer.setGradientEpsilon(1e-4f);

	minimizer.cpu_lbfgs(xstart);

	for (size_t i = 0; i < size; ++i)
	{
		float e = std::abs(minimum[i] - xstart[i]);

		if (e > epsilon)
		{
			cerr << "Is: " << xstart[i] << ", should: " << minimum[i] << endl;
			cerr << "Error " << e << " > epsilon (" << epsilon << "), size = " << size << endl;
			return false;
		}
	}

	return true;
}

int main (int argc, char const *argv[])
{
	if (!test(  2, mat2,   linear2,   5.0f, minimum2,   1e-3f)) exit(EXIT_FAILURE); // 0.1% error allowed
	if (!test(  3, mat3,   linear3,   5.0f, minimum3,   1e-3f)) exit(EXIT_FAILURE); // 0.1% error allowed
	if (!test(  4, mat4,   linear4,   5.0f, minimum4,   1e-3f)) exit(EXIT_FAILURE); // 0.1% error allowed
	if (!test(  5, mat5,   linear5,   5.0f, minimum5,   1e-3f)) exit(EXIT_FAILURE); // 0.1% error allowed
	if (!test( 10, mat10,  linear10,  5.0f, minimum10,  1e-3f)) exit(EXIT_FAILURE); // 0.1% error allowed
	if (!test( 50, mat50,  linear50,  5.0f, minimum50,  3e-2f)) exit(EXIT_FAILURE); // 3.0% error allowed
	if (!test(100, mat100, linear100, 5.0f, minimum100, 3e-2f)) exit(EXIT_FAILURE); // 3.0% error allowed
	if (!test(500, mat500, linear500, 5.0f, minimum500, 3e-2f)) exit(EXIT_FAILURE); // 3.0% error allowed

	cout << "Tests successful." << endl;

	return EXIT_SUCCESS;
}
