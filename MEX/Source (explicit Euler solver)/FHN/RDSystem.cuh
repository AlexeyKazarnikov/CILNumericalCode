#pragma once

#ifdef __CUDACC__
	#include "cuda_runtime.h"
	#include "device_launch_parameters.h"
#else
#define __device__
#endif

template<typename value_type> struct DeviceModelParameters
{
	value_type nu1 = 0;
	value_type nu2 = 0;
	value_type mu = 0;
	value_type eps = 0;
	value_type A = 0;
	value_type B = 0;

	const char* ParameterNames[6]{ "nu1", "nu2", "mu", "eps", "A", "B" };
	const size_t ParameterCount = 6;

	void Initialize(value_type* data)
	{
		this->nu1 = data[0];
		this->nu2 = data[1];
		this->mu = data[2];
		this->eps = data[3];
		this->A = data[4];
		this->B = data[5];
	}
};

template<typename value_type> __device__ inline value_type f(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return par.eps * (v - par.B * u - par.A);
}
template<typename value_type> __device__ inline value_type g(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return -u + v * (par.mu - v * v);
}