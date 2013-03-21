#include "lbfgs.h"
#include "matrices.h"

#include <iostream>
#include <cstdlib>
#include <cmath>
using namespace std;

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
	gpu_quadratic(size_t n, const float *A, const float *b, const float c)
	: cost_function(n)
	{
		CudaSafeCall( cudaMalloc(&m_d_A, n*n * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_b,   n * sizeof(float)) );
		CudaSafeCall( cudaMalloc(&m_d_c,   1 * sizeof(float)) );

		CudaSafeCall( cudaMemcpy(m_d_A,  A, n*n * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall( cudaMemcpy(m_d_b,  b,   n * sizeof(float), cudaMemcpyHostToDevice) );
		CudaSafeCall(  cudaMemcpy(m_d_c, &c,   1 * sizeof(float), cudaMemcpyHostToDevice));

		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tmp1, gpu_quadratic_d::tmp1) );
		CudaSafeCall( cudaGetSymbolAddress((void**)&m_d_tmp2, gpu_quadratic_d::tmp2) );
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

		CudaSafeCall( cudaMemset(m_d_tmp1, 0, sizeof(float)) );
		CudaSafeCall( cudaMemset(m_d_tmp2, 0, sizeof(float)) );

		CudaCheckError();

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

bool test(const size_t size, const float *matrix, const float *linear, const float constant,
	const float *minimum, const float epsilon)
{
	float *xstart = new float[size];
	srand(0xDEADBEEF);

	for (size_t i = 0; i < size; ++i)
	{
		xstart[i] = minimum[i] + 40.0f * float(rand()) / float(RAND_MAX) - 20.0f;
	}

	float *d_x;
	CudaSafeCall( cudaMalloc((void**)&d_x, size * sizeof(float)) );
	CudaSafeCall( cudaMemcpy(d_x, xstart, size * sizeof(float), cudaMemcpyHostToDevice) );

	gpu_quadratic cost(size, matrix, linear, constant);
	lbfgs minimizer(cost);
	minimizer.setGradientEpsilon(1e-4f);

	minimizer.minimize(d_x);

	CudaSafeCall( cudaMemcpy(xstart, d_x, size * sizeof(float), cudaMemcpyDeviceToHost) );
	CudaSafeCall( cudaFree(d_x) );

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
	if (!test( 50, mat50,  linear50,  5.0f, minimum50,  5e-2f)) exit(EXIT_FAILURE); // 5.0% error allowed
	if (!test(100, mat100, linear100, 5.0f, minimum100, 5e-2f)) exit(EXIT_FAILURE); // 5.0% error allowed
	if (!test(500, mat500, linear500, 5.0f, minimum500, 5e-2f)) exit(EXIT_FAILURE); // 5.0% error allowed

	cout << "Tests successful." << endl;

	return EXIT_SUCCESS;
}
