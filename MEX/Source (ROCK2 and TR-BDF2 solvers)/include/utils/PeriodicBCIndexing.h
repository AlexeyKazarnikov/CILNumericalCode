#pragma once

#ifndef __CUDACC__
#define __host__ 
#define __device__
#endif

inline unsigned int __host__ __device__ prev_index(const unsigned int k, const unsigned int Ndim, const unsigned int span = 1)
{
	return (k >= span) * (k - span) + (k < span) * (Ndim + k - span);
}

inline unsigned int __host__ __device__ next_index(const unsigned int k, const unsigned int Ndim, const unsigned int span = 1)
{
	return (k + span < Ndim) * (k + span) + (k + span >= Ndim) * (k + span - Ndim);
}
