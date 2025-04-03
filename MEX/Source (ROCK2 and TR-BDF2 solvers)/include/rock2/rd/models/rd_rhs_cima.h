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
	value_type k1;
	value_type k2;
	value_type k3;
};

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr)
{
	arr[0] = par.alpha;
	arr[1] = par.k1;
	arr[2] = par.k2;
	arr[3] = par.k3;
}

template <typename value_type>
__inline__ __device__ value_type f(const value_type v, const value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return k1 - k2 * v - 4 * k3 * (v * w) / (alpha + v * v);
}

template <typename value_type>
__inline__ __device__ value_type g(const value_type v, const value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return k2 * v - k3 * (v * w) / (alpha + v * v);
}

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type v, value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return  -k2 - 4 * k3 * (alpha * w - v * v * w) / ((alpha + v * v) * (alpha + v * v));
}

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type v, value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return -4 * k3 * v / (alpha + v * v);
}


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type v, value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return k2 - k3 * (alpha * w - v * v * w) / ((alpha + v * v) * (alpha + v * v));
}

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type v, value_type w, const value_type* par)
{
	value_type alpha = par[0];
	value_type k1 = par[1];
	value_type k2 = par[2];
	value_type k3 = par[3];
	
	return -k3 * v / (alpha + v * v);
}
