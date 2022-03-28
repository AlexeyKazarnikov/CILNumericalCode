#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// warp-level sum reduction
template <typename value_type> __inline__ __device__ value_type warp_reduce_sum(value_type val)
{
	val += __shfl_down_sync(0xffffffff, val, 16, 32);
	val += __shfl_down_sync(0xffffffff, val, 8, 32);
	val += __shfl_down_sync(0xffffffff, val, 4, 32);
	val += __shfl_down_sync(0xffffffff, val, 2, 32);
	val += __shfl_down_sync(0xffffffff, val, 1, 32);
	return val;
}

// warp-level max reduction
template <typename value_type> __inline__ __device__ value_type warp_reduce_max(value_type val)
{
	val = max(val, __shfl_down_sync(0xffffffff, val, 16, 32));
	val = max(val, __shfl_down_sync(0xffffffff, val, 8, 32));
	val = max(val, __shfl_down_sync(0xffffffff, val, 4, 32));
	val = max(val, __shfl_down_sync(0xffffffff, val, 2, 32));
	val = max(val, __shfl_down_sync(0xffffffff, val, 1, 32));
	return val;
}