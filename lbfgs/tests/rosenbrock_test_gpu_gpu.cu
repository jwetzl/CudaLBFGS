#include "lbfgs.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

namespace gpu_rosenbrock_d
{
	__global__ void kernelF(const float *d_x, float *d_y)
	{
		const float &x0 = d_x[0];
		const float &x1 = d_x[1];

		// f = (1-x0)^2 + 100 (x1-x0^2)^2

		const float a = (1.0 - x0);
		const float b = (x1 - x0 * x0) ;

		*d_y = (a*a) + 100.0f * (b*b);
	}

	__global__ void kernelGradf(const float *d_x, float *d_grad)
	{
		const float x0 = d_x[0];
		const float x1 = d_x[1];

		// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
		// df/dx1 = 200 (x1 - x0^2)

		d_grad[0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
		d_grad[1] = 200.0f * (x1 - x0*x0);
	}
}

class gpu_rosenbrock : public cost_function
{
public:
	gpu_rosenbrock()
	: cost_function(2) {}

	void f(const float *d_x, float *d_y)
	{
		gpu_rosenbrock_d::kernelF<<<1, 1>>>(d_x, d_y);
	}

	void gradf(const float *d_x, float *d_grad)
	{
		gpu_rosenbrock_d::kernelGradf<<<1, 1>>>(d_x, d_grad);
	}

	void f_gradf(const float *d_x, float *d_f, float *d_grad)
	{
		f(d_x, d_f);
		gradf(d_x, d_grad);
	}
};

bool test(floatdouble x0, floatdouble x1, floatdouble epsilon)
{
	float xstart[] = { float(x0), float(x1) };
	float *d_x;
	cudaMalloc((void**)&d_x, 2 * sizeof(float));
	cudaMemcpy(d_x, xstart, 2 * sizeof(float), cudaMemcpyHostToDevice);

	gpu_rosenbrock rcf;
	lbfgs minimizer(rcf);
	minimizer.setGradientEpsilon(1e-3f);

	lbfgs::status stat = minimizer.minimize(d_x);

	cudaMemcpy(xstart, d_x, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_x);

	floatdouble e0 = std::abs(xstart[0] - 1.0f);
	floatdouble e1 = std::abs(xstart[1] - 1.0f);

	if (e0 > epsilon || e1 > epsilon)
	{
		cerr << "Ended because: " << minimizer.statusToString(stat).c_str() << endl;
		cerr << "Starting point (" << x0 << ", " << x1 << ")" << endl;
		// cerr << "x = " << xstart[0] << ", err(x) = " << e0 << endl;
		// cerr << "y = " << xstart[1] << ", err(y) = " << e1 << endl;
		// cerr << "Max. allowed error: " << epsilon << endl;
		return false;
	}

	return true;
}

int main (int argc, char const *argv[])
{
	for (int i = -4; i < 5; ++i)
	{
		for (int j = -4; j < 5; ++j)
		{
			if (!test(floatdouble(i), floatdouble(j), 1e-2f))
				exit(EXIT_FAILURE);
		}
	}

	cout << "Tests successful." << endl;

	return EXIT_SUCCESS;
}
