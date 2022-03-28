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
	value_type alpha;
	value_type eps;
	value_type mu;
};

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr)
{
	arr[0] = par.eps;
	arr[1] = par.alpha;
	arr[2] = par.mu;
}

template <typename value_type>
__inline__ __device__ value_type f(const value_type u, const value_type v, const value_type* par)
{
	return par[0] * (v - par[1] * u);
}

template <typename value_type>
__inline__ __device__ value_type g(const value_type u, const value_type v, const value_type* par)
{
	return -u + par[2] * v - v * v * v;
}

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type u, value_type v, const value_type* par)
{
	return  -par[0] * par[1];
}

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type u, value_type v, const value_type* par)
{
	return par[0];
}


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type u, value_type v, const value_type* par)
{
	return value_type(-1);
}

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type u, value_type v, const value_type* par)
{
	return par[2] - value_type(3) * v * v;
}
