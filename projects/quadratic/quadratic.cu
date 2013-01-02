#include "CudaLBFGS/lbfgs.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

// f(x)     = x^T A x + b^T x + c
// gradf(x) =   2 A x + b

class cpu_quadratic : public cpu_cost_function
{
public:
	// A is expected to point to n*n floats
	// b is expected to point to n floats
	cpu_quadratic(size_t n, float *A, float *b, float c)
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
	float *m_A;
	float *m_b;
	float  m_c;
};

namespace gpu_quadratic_d
{
	__device__ float tmp1;
	__device__ float tmp2;

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

	__global__ void kernelF(const float *d_xAx, const float *d_bx, const float *d_c, float *d_y)
	{
		*d_y = *d_xAx + *d_bx + *d_c;
	}

	__global__ void kernelGradf(const float *d_x, float *d_grad, float *A, float *b, const size_t len)
	{
		size_t index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= len)
			return;

		d_grad[index] = 0.0f;

		for (size_t j = 0; j < len; ++j)
		{
			d_grad[index] += A[index * len + j] * d_x[j];
		}

		d_grad[index] *= 2.0f;
		d_grad[index] += b[index];
	}

	__global__ static void xAx(const float *x, const float *A, const size_t len, float *res)
	{
		size_t index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= len*len)
			return;

		__shared__ float s_sum; // block local aggregate

		s_sum = 0.0f;

		__syncthreads(); // wait for all to initialize

		const size_t i = index / len;
		const size_t j = index % len;

		myAtomicAdd(&s_sum, A[index] * x[i] * x[j]);

		__syncthreads();

		if (threadIdx.x == 0)
			myAtomicAdd(res, s_sum);
	}

	__global__ static void dot(const float *a, const float *b, const size_t len, float *res)
	{
		size_t index = blockIdx.x * blockDim.x + threadIdx.x;

		if (index >= len)
			return;

		__shared__ float s_sum; // block local aggregate

		s_sum = 0.0f;

		__syncthreads(); // wait for all to initialize

		myAtomicAdd(&s_sum, a[index] * b[index]);

		__syncthreads();

		if (threadIdx.x == 0)
			myAtomicAdd(res, s_sum);
	}

}

class gpu_quadratic : public cost_function
{
public:
	// A is expected to point to n*n floats
	// b is expected to point to n floats
	gpu_quadratic(size_t n, float *A, float *b, float c)
	: cost_function(n)
	{
		CudaSafeCall( cudaMalloc(&m_d_A, n*n * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_b,   n * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_c,   1 * sizeof(float)) );

		CudaSafeCall( cudaMemcpy(m_d_A,  A, n*n * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_b,  b,   n * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_c, &c,   1 * sizeof(float), cudaMemcpyHostToDevice));

		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tmp1, "gpu_quadratic_d::tmp1") );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tmp2, "gpu_quadratic_d::tmp2") );
	}

	~gpu_quadratic()
	{
		CudaSafeCall( cudaFree(m_d_A) );
		CudaSafeCall( cudaFree(m_d_b) );
		CudaSafeCall( cudaFree(m_d_c) );
	}

	void f(const float *d_x, float *d_y)
	{
		const size_t &NX = m_numDimensions;

		dim3 blockDim(512);

		size_t NX2 = NX * NX;

		dim3 gridDim_xAx((NX2 % blockDim.x) == 0 ? (NX2 / blockDim.x)
												 : (NX2 / blockDim.x) + 1);

		dim3 gridDim_dot((NX % blockDim.x) == 0 ? (NX / blockDim.x)
												: (NX / blockDim.x) + 1);

		cudaMemset(m_d_tmp1, 0, sizeof(float));
		cudaMemset(m_d_tmp2, 0, sizeof(float));

		gpu_quadratic_d::xAx<<<gridDim_xAx, blockDim>>>(d_x, m_d_A, NX, m_d_tmp1);
		CudaCheckError();

		gpu_quadratic_d::dot<<<gridDim_dot, blockDim>>>(d_x, m_d_b, NX, m_d_tmp2);
		CudaCheckError();

		cudaDeviceSynchronize();

		gpu_quadratic_d::kernelF<<<1, 1>>>(m_d_tmp1, m_d_tmp2, m_d_c, d_y);
		CudaCheckError();
	}

	void gradf(const float *d_x, float *d_grad)
	{
		const size_t &NX = m_numDimensions;
		dim3 blockDim(512);
		dim3 gridDim((NX % blockDim.x) == 0 ? (NX / blockDim.x)
											: (NX / blockDim.x) + 1);

		gpu_quadratic_d::kernelGradf<<<gridDim, blockDim>>>(d_x, d_grad, m_d_A, m_d_b, NX);
		CudaCheckError();
	}

	void f_gradf(const float *d_x, float *d_f, float *d_grad)
	{
		f(d_x, d_f);
		gradf(d_x, d_grad);
	}

private:

	float *m_d_A;
	float *m_d_b;
	float *m_d_c;
	float *m_d_tmp1;
	float *m_d_tmp2;
};

int main(int argc, char **argv)
{
	// CPU

	size_t n = /*2*/ 8 /*200*/ /*500*/ /*5000*/;
	size_t maxIter = 500;
	float gradientEps = 1e-3f;

	float *A = new float[n*n];

	for (size_t i = 0; i < n; ++i)
	{
		for (size_t j = 0; j < n; ++j)
		{
			//A[i * n + j] = (i == j) ? 1.0f : 0.0f;

			// 8 on main diagonal, 1 on side diagonals, 0 else
			if (i == j)
				A[i * n + j] = 8.0f;
			else if ((i == j-1) || (i == j+1))
				A[i * n + j] = 1.0f;
			else
				A[i * n + j] = 0.0f;
		}
	}

	float *b = new float[n];

	for (size_t i = 0; i < n; ++i)
	{
		b[i] = 1.0f;
	}

	float c = 42.0f;

	cpu_quadratic p1(n, A, b, c);
	lbfgs minimizer1(p1);
	minimizer1.setMaxIterations(maxIter);
	minimizer1.setGradientEpsilon(gradientEps);

	float *x = new float[n];

	for (size_t i = 0; i < n; ++i)
	{
		x[i] = i % 2 == 0 ? 5.0f : -10.0f;
	}

	lbfgs::status stat = minimizer1.minimize_with_host_x(x);

	cout << lbfgs::statusToString(stat).c_str() << endl;

	cout << "CPU quadratic:";

	for (size_t i = 0; i < n; ++i)
	{
		cout << " " << x[i];
	}

	cout << endl;


	// GPU

	for (size_t i = 0; i < n; ++i)
	{
		x[i] = i % 2 == 0 ? 5.0f : -10.0f;
	}

	gpu_quadratic p2(n, A, b, c);
	lbfgs minimizer2(p2);
	minimizer2.setMaxIterations(maxIter);
	minimizer2.setGradientEpsilon(gradientEps);

	float *d_x;
	CudaSafeCall( cudaMalloc(&d_x,   n * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice) );

	lbfgs::status stat2 = minimizer2.minimize(d_x);

	cout << lbfgs::statusToString(stat2).c_str() << endl;

	CudaSafeCall( cudaMemcpy(x, d_x, n * sizeof(float), cudaMemcpyDeviceToHost) );

	cout << "GPU quadratic:";

	for (size_t i = 0; i < n; ++i)
	{
		cout << " " << x[i];
	}

	cout << endl;

	delete [] x;
	delete [] A;
	delete [] b;

	return 0;
}

