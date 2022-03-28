/* 
*/

#pragma once

// These macro should be defined to allow using the routines in both CPU-based and GPU-based code.
#ifndef __CUDACC__
#define __inline__
#define __host__ 
#define __device__
#endif

template <typename value_type> struct ModelParameters
{
	value_type m1;
	value_type m2;
	value_type m3;
	value_type k;
};

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr)
{
	arr[0] = par.m1;
	arr[1] = par.m2;
	arr[2] = par.m3;
	arr[3] = par.k;
}

template <typename value_type>
__inline__ __device__ value_type f(const value_type u, const value_type v, const value_type* par)
{
	return -u - u * v + par[0] * u * u / (1 + par[3] * u * u);
}

template <typename value_type>
__inline__ __device__ value_type g(const value_type u, const value_type v, const value_type* par)
{
	return -par[2] * v - u * v + par[1] * u * u  / (1 + par[3] * u * u);
}

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type u, value_type v, const value_type* par)
{
	return (2 * par[0] * u) / (par[3] * u * u + 1) - v - (2 * par[3] * par[0] * u * u * u) / ((par[3] * u * u + 1) * (par[3] * u * u + 1)) - 1;
}

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type u, value_type v, const value_type* par)
{
	return -u;
}


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type u, value_type v, const value_type* par)
{
	return (2 * par[1] * u) / (par[3] * u * u + 1) - v - (2 * par[3] * par[1] * u * u * u) / ((par[3] * u * u + 1) * (par[3] * u * u + 1));
}

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type u, value_type v, const value_type* par)
{
	return -par[2] - u;
}
