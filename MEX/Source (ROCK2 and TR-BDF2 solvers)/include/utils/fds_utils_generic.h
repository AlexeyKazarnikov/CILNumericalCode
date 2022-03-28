/* This header file contains generic definitions of the finite difference operations.
* Concrete implementations of the respective routines are provided in the files, which
* are specific for boundary conditions.
*/

#pragma once

// These macro should be defined to allow using the routines in both CPU-based and GPU-based code.
#ifndef __CUDACC__
#define __host__ 
#define __device__
#endif

#include "mc_models/shift_handler.cuh"

inline unsigned int __host__ __device__ prev_index(
	const unsigned int k,
	const unsigned int Ndim,
	const unsigned int span
);

inline unsigned int __host__ __device__ next_index(
	const unsigned int k,
	const unsigned int Ndim,
	const unsigned int span
);

template <typename value_type, typename index_type> value_type __host__ __device__ get_next_element(
	const shift_handler<value_type, index_type>& u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type> value_type __host__ __device__ get_next_element(
	const value_type* u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type, typename index_type> value_type __host__ __device__ get_after_next_element(
	const shift_handler<value_type, index_type>& u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type> value_type __host__ __device__ get_after_next_element(
	const value_type* u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type, typename index_type> value_type __host__ __device__ get_prev_element(
	const shift_handler<value_type, index_type>& u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type> value_type __host__ __device__ get_prev_element(
	const value_type* u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type, typename index_type> value_type __host__ __device__ get_after_prev_element(
	const shift_handler<value_type, index_type>& u,
	const unsigned int k,
	const unsigned int Ndim
);

template <typename value_type> value_type __host__ __device__ get_after_prev_element(
	const value_type* u,
	const unsigned int k,
	const unsigned int Ndim
);

template<typename value_type> inline __host__ __device__ value_type diffFDS(
	const value_type u_k,
	const value_type u_m,
	const value_type dxinv
)
{
	return (u_k - u_m) * dxinv;
}

template<typename value_type> inline __host__ __device__ value_type diff2FDS(
	const value_type u_k,
	const value_type u_p,
	const value_type u_m,
	const value_type dx2inv
)
{
	return (u_p - 2 * u_k + u_m) * dx2inv;
}

template<typename value_type> inline __host__ __device__ value_type diff3FDS(
	const value_type u_k,
	const value_type u_p,
	const value_type u_m,
	const value_type u_pp,
	const value_type u_mm,
	const value_type dx3inv
)
{
	return (u_pp - 2 * u_p + 2 * u_m - u_mm) * (value_type(0.5) * dx3inv);
}

template<typename value_type> inline __host__ __device__ value_type diff4FDS(
	const value_type u_k,
	const value_type u_p,
	const value_type u_m,
	const value_type u_pp,
	const value_type u_mm,
	const value_type dx4inv
)
{
	return (u_pp - 4 * u_p + 6 * u_k - 4 * u_m + u_mm) * dx4inv;
}

/*

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diffFDS(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim,
	const value_type dxinv
);

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff2FDS(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim,
	const value_type dx2inv
);

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff3FDS(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim,
	const value_type dx3inv
);

template<typename handler_type, typename value_type> inline __host__ __device__ value_type diff4FDS(
	const handler_type& u,
	const unsigned int k,
	const unsigned int Ndim,
	const value_type dx4inv
);

*/