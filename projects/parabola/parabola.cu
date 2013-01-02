#include "CudaLBFGS/lbfgs.h"
#include "CudaLBFGS/error_checking.h"
#include "CudaLBFGS/timer.h"

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

using namespace std;

class cpu_parabola : public cpu_cost_function
{
public:
	cpu_parabola(float a, float b, float c) 
	: cpu_cost_function(1)
	, m_a(a)
	, m_b(b)
	, m_c(c) {}
	
	void cpu_f(const floatdouble *h_x, floatdouble *h_y)
	{
		const float x = *h_x;
		*h_y = m_a * x * x + m_b * x + m_c;
	}
	
	void cpu_gradf(const floatdouble *h_x, floatdouble *h_grad)
	{
		const float x = *h_x;
		*h_grad = 2.0f * m_a * x + m_b;
	}

	void cpu_f_gradf(const floatdouble *h_x, floatdouble *h_f, floatdouble *h_gradf)
	{
		cpu_f(h_x, h_f);
		cpu_gradf(h_x, h_gradf);
	}
	
private:
	float m_a;
	float m_b;
	float m_c;
};

namespace gpu_parabola_d
{
	__device__ float m_a;
	__device__ float m_b;
	__device__ float m_c;

	__global__ void kernel_combined_f_gradf(const float *d_x, float *d_y, float *d_grad)
	{
		const float &x = *d_x;
		*d_y = m_a * x * x + m_b * x + m_c;
		*d_grad = 2.0f * m_a * x + m_b;
	}
}

class gpu_parabola : public cost_function
{
public:
	gpu_parabola(float a, float b, float c) 
	: cost_function(1) 
	{
		cudaGetSymbolAddress((void**)&m_d_a, "gpu_parabola_d::m_a");
		cudaGetSymbolAddress((void**)&m_d_b, "gpu_parabola_d::m_b");
		cudaGetSymbolAddress((void**)&m_d_c, "gpu_parabola_d::m_c");

		cudaMemcpy(m_d_a, &a, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_d_b, &b, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(m_d_c, &c, sizeof(float), cudaMemcpyHostToDevice);
	}
	
	void f_gradf(const float *d_x, float *d_f, float *d_grad)
	{
		gpu_parabola_d::kernel_combined_f_gradf<<<1, 1>>>(d_x, d_f, d_grad);
		CudaCheckError();
	}
	
private:

	float *m_d_a; 
	float *m_d_b;
	float *m_d_c;
};

int main(int argc, char **argv)
{
	// CPU

	cpu_parabola p1(4.0f, 2.0f, 6.0f);
	lbfgs minimizer1(p1);

	float x = 8.0f;
	{
		timer t("parabola_cpu");
		t.start();
		minimizer1.minimize_with_host_x(&x);
	}
	
	cout << "CPU Parabola: " << x << endl;

	// GPU

	gpu_parabola p2(4.0f, 2.0f, 6.0f);
	lbfgs minimizer2(p2);
	
	x = 8.0f;
	
	float *d_x;
	cudaMalloc(&d_x, sizeof(float));
	cudaMemcpy(d_x, &x, sizeof(float), cudaMemcpyHostToDevice);

	{
		timer t("parabola_gpu");
		t.start();
		minimizer2.minimize(d_x);
	}

	cudaMemcpy(&x, d_x, sizeof(float), cudaMemcpyDeviceToHost);

	cout << "GPU Parabola: " << x << endl;

	return 0;
}
