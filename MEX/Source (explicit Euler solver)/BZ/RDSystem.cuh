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
	value_type A = 0;
	value_type B = 0;

	const char* ParameterNames[4]{ "nu1", "nu2", "A", "B" };
	const size_t ParameterCount = 4;

	void Initialize(value_type* data)
	{
		this->nu1 = data[0];
		this->nu2 = data[1];
		this->A = data[2];
		this->B = data[3];
	}
};

template<typename value_type> __device__ inline value_type f(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return par.A - u * (par.B + 1) + u * u * v;
}
template<typename value_type> __device__ inline value_type g(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return u * par.B - u * u * v;
}