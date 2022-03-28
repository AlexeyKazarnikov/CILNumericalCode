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
	value_type mua;
	value_type mui;
	value_type rhoa;
	value_type rhoi;
};

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr)
{
	arr[0] = par.mua;
	arr[1] = par.mui;
	arr[2] = par.rhoa;
	arr[3] = par.rhoi;
}

template <typename value_type>
__inline__ __device__ value_type f(const value_type u, const value_type v, const value_type* par)
{
	return  u * (par[2] * u / v - par[0]);
	//return  u * (par[2] * __fdividef((float)(u), (float)(v)) - par[0]);
}

template <typename value_type>
__inline__ __device__ value_type g(const value_type u, const value_type v, const value_type* par)
{
	return par[3] * u * u - par[1] * v;
}

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type u, value_type v, const value_type* par)
{
	return 2 * par[2] * u / v - par[0];
	//return 2 * par[2] * __fdividef((float)(u), (float)(v)) - par[0];
}

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type u, value_type v, const value_type* par)
{
	return - par[2] * u * u / (v * v);
}


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type u, value_type v, const value_type* par)
{
	return 2 * par[3] * u;
}

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type u, value_type v, const value_type* par)
{
	return - par[1];
}
