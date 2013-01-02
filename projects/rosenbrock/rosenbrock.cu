#include "CudaLBFGS/lbfgs.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

class cpu_rosenbrock : public cpu_cost_function
{
public:
	cpu_rosenbrock()
	: cpu_cost_function(2) {}

	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		const floatdouble x0 = h_x[0];
		const floatdouble x1 = h_x[1];

		// f = (1-x0)^2 + 100 (x1-x0^2)^2

		const floatdouble a = (1.0 - x0);
		const floatdouble b = (x1 - x0 * x0) ;

		*h_y = (a*a) + 100.0f * (b*b);
	}

	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		const floatdouble x0 = h_x[0];
		const floatdouble x1 = h_x[1];

		// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
		// df/dx1 = 200 (x1 - x0^2)

		h_grad[0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
		h_grad[1] = 200.0f * (x1 - x0*x0);
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}
};

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

int main(int argc, char **argv)
{
	// CPU

	cpu_rosenbrock rb1;
	lbfgs minimizer1(rb1);
	minimizer1.setGradientEpsilon(1e-3f);

	float x[2] = {2.0f, -1.0f};
	lbfgs::status stat = minimizer1.cpu_lbfgs(x);

	cout << "CPU Rosenbrock: " << x[0] << " " << x[1] << endl;
	cout << minimizer1.statusToString(stat).c_str() << endl;

	// GPU

	 gpu_rosenbrock rb2;
	 lbfgs minimizer2(rb2);
	
	 x[0] = -4.0f;
	 x[1] = 2.0f;
	
	 float *d_x;
	 cudaMalloc(&d_x,    2 * sizeof(float));
	 cudaMemcpy(d_x, &x, 2 * sizeof(float), cudaMemcpyHostToDevice);
	
	 minimizer2.minimize(d_x);
	
	 cudaMemcpy(&x, d_x, 2 * sizeof(float), cudaMemcpyDeviceToHost);
	
	 cout << "GPU Rosenbrock: " << x[0] << " " << x[1] << endl;

	return 0;
}
