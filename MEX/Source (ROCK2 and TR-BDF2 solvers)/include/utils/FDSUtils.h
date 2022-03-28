#pragma once

#ifndef __CUDACC__
#define __host__ 
#define __device__
#endif


template<typename handler_type, typename value_type> inline __host__ __device__ value_type diffFDS(const handler_type& u, const unsigned int k, const unsigned int km, const value_type dxinv)
{
	return (u[k] - u[km]) * dxinv;
}

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff2FDS(const handler_type& u, const unsigned int k, const unsigned int km, const unsigned int kp, const value_type dx2inv)
{
	return (u[kp] - 2 * u[k] + u[km]) * dx2inv;
}

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff3FDS(const handler_type& u, const unsigned int k, const unsigned int km, const unsigned int kmm, const unsigned int kp, const unsigned int kpp, const value_type dx3inv)
{
	return (u[kpp] - 2 * u[kp] + 2 * u[km] - u[kmm]) * (value_type(0.5) * dx3inv);
}

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff4FDS(const handler_type& u, const unsigned int k, const unsigned int km, const unsigned int kmm, const unsigned int kp, const unsigned int kpp, const value_type dx4inv)
{
	return (u[kpp] - 4 * u[kp] + 6 * u[k] - 4 * u[km] + u[kmm]) * dx4inv;
}
