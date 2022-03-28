#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mc_models/shift_handler.cuh"
#include "utils/fds_utils_generic.h"
#include "utils/reduce_utils.cuh"

template <typename value_type> struct ModelParameters;

// L2-based vector norm
template <typename value_type> __global__ void compute_norm(
	const value_type* w, 
	const size_t N, 
	value_type* norm, 
	const unsigned int* run_indices
)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_w = w + run_indices[blockIdx.x] * N;

	// indexes for the current thread
	const unsigned int k = threadIdx.x;
	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	value_type val = 0;
	if (k < N)
		val = global_w[k] * global_w[k];

	// warp-level reduction
	val = warp_reduce_sum(val);

	// writing reduced values to shared memory
	if (lane == 0) shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (k < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_sum(val); //Final reduce within first warp

	if (k == 0)
	{
		norm[run_indices[blockIdx.x]] = sqrt(val);
	}
}

template <typename value_type> __global__ void compute_diff_norm(
	const value_type* w1, 
	const value_type* w2, 
	const size_t N, 
	value_type* norm, 
	const unsigned int* run_indices
)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_w1 = w1 + run_indices[blockIdx.x] * N;
	const value_type* global_w2 = w2 + run_indices[blockIdx.x] * N;

	// indexes for the current thread
	const unsigned int k = threadIdx.x;
	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	value_type val = 0;
	if (k < N)
		val = (global_w1[k] - global_w2[k]) * (global_w1[k] - global_w2[k]);

	// warp-level reduction
	val = warp_reduce_sum(val);

	// writing reduced values to shared memory
	if (lane == 0) shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (k < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_sum(val); //Final reduce within first warp

	if (k == 0)
	{
		norm[run_indices[blockIdx.x]] = sqrt(val);
		//printf("val = %f", val);
		//printf("norm = %f", norm[run_indices[blockIdx.x]]);
	}
}


template <typename value_type> __global__ void model_arc_Length(
	const value_type* w,
	value_type* arc,
	const unsigned int* run_indices,
	const unsigned int sys_size,
	const value_type dx,
	const value_type dxinv
)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_u = w + run_indices[blockIdx.x] * sys_size; // component for lagrangian multiplier is also taken into account

	// indexes for the current thread
	const unsigned int k = threadIdx.x;

	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	const value_type u_k = global_u[k];
	const value_type u_k_m = get_prev_element(global_u, k, blockDim.x);
	const value_type du = diffFDS(u_k, u_k_m, dxinv);

	value_type val = sqrt(1 + du * du);

	// warp-level reduction
	val = warp_reduce_sum(val);

	// writing reduced values to shared memory
	if (lane == 0) shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (k < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_sum(val); //Final reduce within first warp

	if (k == 0)
		arc[run_indices[blockIdx.x]] = dx * val;
}

// this function evaluates global arc length of the model
template <typename value_type> void evaluate_arc_length(
	const cudaStream_t t_cuda_stream,
	const value_type* t_device_w_m,
	const unsigned int t_sys_size,
	const ModelParameters<value_type> t_par,
	const unsigned int* t_device_run_indices,
	const unsigned int t_num_run,
	value_type* t_device_arc_m
)
{
	dim3 grid{ t_num_run,1,1 };
	dim3 block{ t_par.N,1,1 };
	model_arc_Length << <grid, block, 0, t_cuda_stream >> > (t_device_w_m, t_device_arc_m, t_device_run_indices, t_par.sys_size, t_par.dx, t_par.dxinv);
}

template <typename value_type> __global__ void evaluateJac_p2(
	const value_type* w_m,
	const value_type* dt,
	const unsigned int* run_indices,
	const ModelParameters<value_type> par,
	const value_type jac_step,
	const value_type gamma,
	value_type* jac)
{
	unsigned int k = threadIdx.x;

	const value_type* global_u = w_m + run_indices[blockIdx.x] * par.sys_size; // component for lagrangian multiplier is also taken into account
	const shift_handler<value_type, unsigned int> global_u_h(global_u, k, jac_step);

	const value_type u_k = global_u[k];
	const value_type u_k_m = get_prev_element(global_u, k, par.N);
	const value_type u_k_p = get_next_element(global_u, k, par.N);

	const value_type du = diffFDS(u_k, u_k_m, par.dxinv);
	const value_type du_next = diffFDS(u_k_p, u_k, par.dxinv);
	const value_type d2u = diff2FDS(u_k, u_k_p, u_k_m, par.dx2inv);

	const value_type t2inv = value_type(1) / (1 + du * du);
	const value_type t3inv = t2inv * sqrt(t2inv);

	const value_type u_h_k = global_u_h[k];
	const value_type u_h_p = get_next_element(global_u_h, k, par.N);
	const value_type u_h_m = get_prev_element(global_u_h, k, par.N);

	const value_type du_h = diffFDS(u_h_k, u_h_m, par.dxinv);
	const value_type du_next_h = diffFDS(u_h_p, u_h_k, par.dxinv);

	// printf("k = %i, u[k] = %f, u[k+1] = %f, uh[k] = %f, uh[k+1] = %f \n", k, global_u[k], global_u[kp], global_u_h[k], global_u_h[kp]);

	jac[par.sys_size * par.sys_size * run_indices[blockIdx.x] + par.sys_size * k + par.rhs_size] =
		(par.dx / jac_step) * (
			sqrt(1 + du_h * du_h) + sqrt(1 + du_next_h * du_next_h)
			- sqrt(1 + du * du) - sqrt(1 + du_next * du_next)
			);
	// par.scale_const * !!!
	jac[par.sys_size * par.sys_size * run_indices[blockIdx.x] + par.sys_size * par.rhs_size + k] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] * d2u * t3inv;

	//if (k == 0)
		//printf("bid = %i, ind = %i \n", blockIdx.x, run_indices[blockIdx.x]);
	//	printf("k = %i, km = %i, du = %f, du_next = %f, du_h = %f, du_next_h = %f \n", k, km, du, du_next, du_h, du_next_h);
}