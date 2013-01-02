#include "CudaLBFGS/lbfgs.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

class cpu_rosenbrock_nd : public cpu_cost_function
{
public:
	cpu_rosenbrock_nd(size_t n)
	: cpu_cost_function(n) {

		if (n % 2 != 0)
		{
			std::cerr << "Generalized Rosenbrock is only defined for even number of unknowns." << std::endl;
			std::exit(-1);
		}

	}

	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		*h_y = 0.0f;

		for (size_t i = 0; i < m_numDimensions / 2; ++i)
		{
			const floatdouble x0 = h_x[2*i+0];
			const floatdouble x1 = h_x[2*i+1];

			// f = (1-x0)^2 + 100 (x1-x0^2)^2

			const floatdouble a = (1.0 - x0);
			const floatdouble b = (x1 - x0 * x0) ;

			*h_y += (a*a) + 100.0f * (b*b);
		}
	}

	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		for (size_t i = 0; i < m_numDimensions / 2; ++i)
		{
			const floatdouble x0 = h_x[2*i+0];
			const floatdouble x1 = h_x[2*i+1];

			// df/dx0 = -2 (1-x0) - 400 (x1-x0^2) x0
			// df/dx1 = 200 (x1 - x0^2)

			h_grad[2*i+0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
			h_grad[2*i+1] = 200.0f * (x1 - x0*x0);
		}
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}
};

namespace gpu_rosenbrock_d
{
	__device__ static void myAtomicAdd(float *address, float value)
	{
	#if __CUDA_ARCH__ >= 200
		atomicAdd(address, value);
	#else
		// cf. https://www.sharcnet.ca/help/index.php/CUDA_tips_and_tricks
		int oldval, newval, readback;

		oldval = __float_as_int(*address);
		newval = __float_as_int(__int_as_float(oldval) + value);
		while ((readback=atomicCAS((int *)address, oldval, newval)) != oldval)
		{
			oldval = readback;
			newval = __float_as_int(__int_as_float(oldval) + value);
		}
	#endif
	}

	__global__ void kernelF(const float *d_x, float *d_y, float *d_grad, size_t len)
	{
		size_t index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= len)
			return;

		float res = 0.0f;

		//for (size_t i = 0; i < batch; i += 2)
		{
			const float x0 = d_x[2*index+0];
			const float x1 = d_x[2*index+1];

			// f = (1-x0)^2 + 100 (x1-x0^2)^2

			const float a = (1.0 - x0);
			const float b = (x1 - x0 * x0);

			res += (a*a) + 100.0f * (b*b);

			d_grad[2*index+0] = -2.0f * (1.0f - x0) - 400.0f * x0 * (x1 - x0*x0);
			d_grad[2*index+1] = 200.0f * (x1 - x0*x0);
		}

		__shared__ float s_sum; // block local aggregate

		s_sum = 0.0f;

		__syncthreads(); // wait for all to initialize

		myAtomicAdd(&s_sum, res);

		__syncthreads();

		if (threadIdx.x == 0)
			myAtomicAdd(d_y, s_sum);
	}

}

class gpu_rosenbrock_nd : public cost_function
{
public:
	gpu_rosenbrock_nd(size_t n)
	: cost_function(n) {

		if (n % 2 != 0)
		{
			std::cerr << "Generalized Rosenbrock is only defined for even number of unknowns." << std::endl;
			std::exit(-1);
		}

		m_numBatch = 4;
		while (n % m_numBatch != 0)
			m_numBatch >>= 1;

	}

	void f_gradf(const float *d_x, float *d_f, float *d_grad)
	{
		size_t launches = m_numDimensions / 2;

		dim3 blockDim(512);
		dim3 gridDim((launches % blockDim.x) == 0 ? (launches / blockDim.x)
		                                          : (launches / blockDim.x) + 1);

		const float zero = 0.0f;
		cudaMemcpy(d_f, &zero, sizeof(float), cudaMemcpyHostToDevice);
		gpu_rosenbrock_d::kernelF<<<gridDim, blockDim>>>(d_x, d_f, d_grad, launches);
		cudaDeviceSynchronize();
	}

private:
	size_t m_numBatch;
};

int main(int argc, char **argv)
{
	// CPU

	const size_t NX = atoi(argv[1]);

	cpu_rosenbrock_nd rb1(NX);
	lbfgs minimizer1(rb1);
	minimizer1.setGradientEpsilon(atof(argv[2]));

	lbfgs::status stat;

	float *x = new float[NX];

	for (size_t i = 0; i < NX; ++i)
	{
		x[i] = i % 2 == 1 ? -1 : 2;
	}

	stat = minimizer1.minimize_with_host_x(x);

	cout << "CPU Rosenbrock: ";

	for (size_t i = 0; i < NX-1; ++i)
	{
		cout << x[i] << ", ";
	}

	cout << x[NX-1] << endl;
	cout << minimizer1.statusToString(stat).c_str() << endl;

	// GPU

	gpu_rosenbrock_nd rb2(NX);
	lbfgs minimizer2(rb2);
	minimizer2.setGradientEpsilon(atof(argv[2]));

	for (size_t i = 0; i < NX; ++i)
	{
		x[i] = i % 2 == 1 ? -1 : 2;
	}

	float *d_x;
	cudaMalloc(&d_x,    NX * sizeof(float));
	cudaMemcpy(d_x, x, NX * sizeof(float), cudaMemcpyHostToDevice);

	stat = minimizer2.minimize(d_x);

	cudaMemcpy(x, d_x, NX * sizeof(float), cudaMemcpyDeviceToHost);

	cout << "GPU Rosenbrock: ";

	for (size_t i = 0; i < NX-1; ++i)
	{
		cout << x[i] << ", ";
	}

	cout << x[NX-1] << endl;
	cout << minimizer2.statusToString(stat).c_str() << endl;

	delete [] x;
	cudaFree(d_x);

	return 0;
}
