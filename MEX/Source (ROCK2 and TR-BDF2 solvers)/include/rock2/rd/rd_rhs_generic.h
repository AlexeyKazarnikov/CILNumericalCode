/* 
*/

#pragma once

// These macro should be defined to allow using the routines in both CPU-based and GPU-based code.
#ifndef __CUDACC__
#define __inline__
#define __host__ 
#define __device__
#endif

template <typename value_type> struct ModelParameters;

template <typename value_type> void par_to_array(const ModelParameters<value_type>& par, value_type* arr);

template <typename value_type>
__inline__ __device__ value_type f(const value_type u, const value_type v, const value_type* par);

template <typename value_type>
__inline__ __device__ value_type g(const value_type u, const value_type v, const value_type* par);

template <typename value_type>
__inline__ __device__ value_type dfdu(value_type u, value_type v, const value_type* par);

template <typename value_type>
__inline__ __device__ value_type dfdv(value_type u, value_type v, const value_type* par);


template <typename value_type>
__inline__ __device__ value_type dgdu(value_type u, value_type v, const value_type* par);

template <typename value_type>
__inline__ __device__ value_type dgdv(value_type u, value_type v, const value_type* par);
