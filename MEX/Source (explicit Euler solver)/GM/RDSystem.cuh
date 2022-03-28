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
	value_type mua = 0;
	value_type mui = 0;
	value_type rhoa = 0;
	value_type rhoi = 0;

	const char* ParameterNames[6]{ "nu1", "nu2", "rhoa", "mua", "rhoi", "mui" };
	const size_t ParameterCount = 6;

	void Initialize(value_type* data)
	{
		this->nu1 = data[0];
		this->nu2 = data[1];
		this->rhoa = data[2];
		this->mua = data[3];
		this->rhoi = data[4];
		this->mui = data[5];
	}
};

template<typename value_type> __device__ inline value_type f(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return  u * (par.rhoa * u / v - par.mua);
}
template<typename value_type> __device__ inline value_type g(value_type u, value_type v, DeviceModelParameters<value_type> par)
{
	return par.rhoi * u*u - par.mui * v;
}