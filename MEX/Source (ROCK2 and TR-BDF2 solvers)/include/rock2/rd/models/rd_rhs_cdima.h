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
	value_type sigma;
	value_type a;
	value_type b;
};

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr)
{
	arr[0] = par.sigma;
	arr[1] = par.a;
	arr[2] = par.b;
}

template <typename value_type>
__inline__ __device__ value_type f(const value_type v, const value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return (1 / sigma) * (a - v - 4 * (v * w) / (1 + v * v));
}

template <typename value_type>
__inline__ __device__ value_type g(const value_type v, const value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return b * (v - (v * w) / (1 + v * v));
}

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type v, value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return  -((4 * w) / (v * v + 1) - (8 * v * v * w)/((v * v + 1) * (v * v + 1)) + 1) / sigma;
}

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type v, value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return -(4 * v) / (sigma * (v * v + 1));
}


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type v, value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return b * ((2 * v * v * w) / ((v * v + 1) * (v * v + 1)) - w / (v * v + 1) + 1);
}

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type v, value_type w, const value_type* par)
{
	value_type sigma = par[0];
	value_type a = par[1];
	value_type b = par[2];
	
	return -(b * v) / (v * v + 1);
}
