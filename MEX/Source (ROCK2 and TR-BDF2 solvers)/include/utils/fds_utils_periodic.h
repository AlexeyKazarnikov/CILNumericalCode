/* This header file contains the implementation of finite difference operations for Neumann boundary conditions.
*/

#pragma once

// These macro should be defined to allow using the routines in both CPU-based and GPU-based code.
#ifndef __CUDACC__
#define __host__ 
#define __device__
#endif

inline unsigned int __host__ __device__ prev_index(
	const unsigned int k,
	const unsigned int Ndim,
	const unsigned int span = 1
)
{
	return (k >= span) * (k - span) + (k < span) * (Ndim + k - span);
}

inline unsigned int __host__ __device__ next_index(
	const unsigned int k,
	const unsigned int Ndim,
	const unsigned int span = 1
)
{
	return (k + span < Ndim) * (k + span) + (k + span >= Ndim) * (k + span - Ndim);
}

template <typename handler_type> auto __host__ __device__ get_next_element(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim
)
{
	return u[next_index(k, Ndim)];
}

template <typename handler_type> auto __host__ __device__ get_after_next_element(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim
)
{
	return u[next_index(k, Ndim, 2)];
}

template <typename handler_type> auto __host__ __device__ get_prev_element(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim
)
{
	return u[prev_index(k, Ndim)];
}

template <typename handler_type> auto __host__ __device__ get_after_prev_element(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim
)
{
	return u[prev_index(k, Ndim, 2)];
}
