#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "reduce_utils.cuh"
#include "shift_handler.cuh"

#include "D:\\Programming\\Visual Studio 2017\\Utils\\FDSUtils.h"
#include "D:\\Programming\\Visual Studio 2017\\Utils\\PeriodicBCIndexing.h"

// structure, containing all model parameters. Used only in the following model-specific functions.
template <typename value_type> struct ModelParameters
{
public:
	const value_type alpha;
	const value_type beta;
	const value_type delta;
	const value_type L;
	const value_type D;
	const value_type tau;
	const unsigned int N;
	const value_type dx;
	const value_type dxinv;
	const value_type dx2inv;
	const value_type dx3inv;
	const value_type dx4inv;
	const value_type scale_const;
	const unsigned int sys_dim = 2;
	const unsigned int sys_size;
	const unsigned int rhs_size;

	ModelParameters(
		value_type t_alpha,
		value_type t_beta,
		value_type t_delta,
		value_type t_L,
		value_type t_D,
		value_type t_tau,
		unsigned int t_N)
		:
		alpha(t_alpha),
		beta(t_beta),
		delta(t_delta),
		L(t_L),
		D(t_D),
		tau(t_tau),
		N(t_N),
		dx(this->L / this->N),
		dxinv(this->N / this->L),
		dx2inv(dxinv* dxinv),
		dx3inv(dx2inv* dxinv),
		dx4inv(dx2inv* dx2inv),
		scale_const(-this->L / this->tau),
		sys_size(2 * N + 1),
		rhs_size(2 * N)
	{}
};

template <typename value_type> __global__ void model_arc_Length(
	const value_type* w, 
	value_type* arc, 
	const unsigned int* run_indices, 
	const unsigned int sys_size, 
	const value_type L)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_u = w + run_indices[blockIdx.x] * sys_size; // component for lagrangian multiplier is also taken into account

	// indexes for the current thread
	const unsigned int k = threadIdx.x;
	const unsigned int km = prev_index(k, blockDim.x);

	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	const value_type du = diffFDS(global_u, k, km, blockDim.x / L);

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
		arc[run_indices[blockIdx.x]] = (L / blockDim.x) * val;
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
	dim3 block{t_par.N,1,1};
	model_arc_Length<<<grid,block,0,t_cuda_stream>>>(t_device_w_m, t_device_arc_m, t_device_run_indices, t_par.sys_size, t_par.L);
}

template <typename handler_type, typename value_type> __device__ void compute_rhs(
	const handler_type& t_global_u,
	const handler_type& t_global_phi,
	const value_type t_lambda,
	const unsigned int t_k,
	const ModelParameters<value_type>& t_par,
	value_type& t_dudt,
	value_type& t_dphidt)
{
	// indexes for the current thread
	unsigned int km = prev_index(t_k, t_par.N);
	unsigned int kmm = prev_index(km, t_par.N);
	unsigned int kp = next_index(t_k, t_par.N);
	unsigned int kpp = next_index(kp, t_par.N);

	// compute chemistry (needed for coupling)
	value_type dphi;
	value_type d2phi;

	// mechanical part evaluation
	value_type du;
	value_type d2u;
	value_type d3u;
	value_type d4u;

	value_type t2inv;
	value_type t4inv;

	value_type t3inv;

	value_type du_d2u;
	value_type du_d2u_t4inv;

	value_type k0;
	value_type dk0;
	value_type d2k0;

	value_type k_hat;
	value_type dk_hat;
	value_type d2k_hat;

	value_type f;

	dphi = diffFDS(t_global_phi, t_k, km, t_par.dxinv);
	d2phi = diff2FDS(t_global_phi, t_k, km, kp, t_par.dx2inv);

	// mechanical part evaluation
	du = diffFDS(t_global_u, t_k, km, t_par.dxinv);
	d2u = diff2FDS(t_global_u, t_k, km, kp, t_par.dx2inv);
	d3u = diff3FDS(t_global_u, t_k, km, kmm, kp, kpp, t_par.dx3inv);
	d4u = diff4FDS(t_global_u, t_k, km, kmm, kp, kpp, t_par.dx4inv);

	t2inv = value_type(1) / (1 + du * du);
	t3inv = t2inv * sqrt(t2inv);
	t4inv = t2inv * t2inv;

	du_d2u = du * d2u;
	du_d2u_t4inv = du_d2u * t4inv;

	k0 = d2u * t3inv;
	dk0 = t3inv * (d3u - 3 * du_d2u * d2u * t2inv);
	d2k0 = t3inv * (d4u - d2u * (3 * t2inv * (d2u * d2u + 3 * du * d3u) - 15 * du_d2u * du_d2u_t4inv));

	k_hat = k0 + t_par.beta * t_global_phi[t_k];
	dk_hat = dk0 + t_par.beta * dphi;
	d2k_hat = d2k0 + t_par.beta * d2phi;

	f = t_par.delta * (k0 < 0) * k0 / (k0 - 1);

	// evaluate the r.h.s.
	t_dudt = t_par.scale_const * (d2k_hat * t2inv - dk_hat * du_d2u_t4inv - k_hat * k0 * k0) + t_lambda * k0;
	t_dphidt = t_par.D * (d2phi * t2inv - dphi * du_d2u_t4inv) - t_par.alpha * t_global_phi[t_k] + f;
}


template <typename value_type> __global__ void model_rhs(
	const value_type* w, 
	value_type* dwdt, 
	const unsigned int* run_indices, 
	const ModelParameters<value_type> par)
{
	// first we determine the current system position in the global memory block
	// note that each thread corresponts to two (!) nodes of the model (surface u(k) and morphogen pki(k)).
	// that is why one block corresponds to 2 * par.Ndim elements in memory
	const value_type* global_u = w + run_indices[blockIdx.x] * (2 * par.N + 1); // component for lagrangian multiplier is also taken into account
	const value_type* global_phi = global_u + par.N;
	const value_type lambda = *(global_phi + par.N);

	value_type* global_dudt = dwdt + run_indices[blockIdx.x] * 2 * par.N; // NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	value_type* global_dphidt = global_dudt + par.N;


	value_type dudt, dphidt;

	// evaluate the r.h.s.
	compute_rhs(global_u, global_phi, lambda, threadIdx.x, par, dudt, dphidt);

	global_dudt[threadIdx.x] = dudt;
	global_dphidt[threadIdx.x] = dphidt;

	//printf("k = %i, u = %f, phi = %f, du = %f, dphi = %f, lambda = %f \n", threadIdx.x, global_u[threadIdx.x], global_phi[threadIdx.x], dudt, dphidt, lambda);
}

// this function evaluates r.h.s. of the model
template <typename value_type> void evaluate_rhs(
	const cudaStream_t t_cuda_stream,
	const value_type* t_device_w_m,
	const unsigned int t_sys_size,
	const ModelParameters<value_type> t_par,
	const unsigned int* t_device_run_indices,
	const unsigned int t_num_run,
	value_type* t_device_rhs_m
)
{
	dim3 grid{ t_num_run,1,1 };
	dim3 block{t_par.N,1,1};
	model_rhs <<<grid,block,0,t_cuda_stream>>>(t_device_w_m, t_device_rhs_m, t_device_run_indices, t_par);
}

template <typename value_type> __global__ void model_rhs_max_norm(
	const value_type* w,
	const size_t N,
	value_type* norm,
	const unsigned int sys_dim,
	const unsigned int dim_size,
	const unsigned int* run_indices)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_w = w + run_indices[blockIdx.x] * sys_dim * dim_size + dim_size;

	// indexes for the current thread
	const unsigned int k = threadIdx.x;
	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	value_type val = 0;
	if (k < N)
		val = abs(global_w[k]);

	// warp-level reduction
	val = warp_reduce_max(val);

	// writing reduced values to shared memory
	if (lane == 0) shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (k < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_max(val); //Final reduce within first warp

	if (k == 0)
	{
		norm[run_indices[blockIdx.x]] = val;
	}
}

// this function computes the norm for the r.h.s. of the model
template <typename value_type> void evaluate_rhs_norm(
	const cudaStream_t t_cuda_stream,
	const value_type* t_device_rhs_m,
	const unsigned int t_sys_size,
	const ModelParameters<value_type> t_par,
	const unsigned int* t_device_run_indices,
	const unsigned int t_num_run,
	value_type* t_device_norm_m
)
{
	unsigned int warp_size = 32;
	unsigned int warp_count = t_par.rhs_size / warp_size + (t_par.rhs_size % warp_size != 0);
	dim3 grid{ t_num_run,1,1 };
	dim3 block{ warp_size * warp_count,1,1 };
	model_rhs_max_norm<<<grid,block,0,t_cuda_stream>>>(t_device_rhs_m, t_par.N, t_device_norm_m, t_par.sys_dim, t_par.N, t_device_run_indices);
}

// the block has the following structure: (i,j) = (N,8), where
// i = [0,N) is equation number
// j = [0,8) is variable number, where 0..4 stand for u and 5..7 stand for phi
// the output is the evaluation of d rhs(u_i) / dv_j and d rhs(phi_i) / dv_j
template <typename value_type> __global__ void evaluateJac_p1(
	const value_type* w_m, 
	const value_type* rhs_m, 
	const value_type* dt, 
	const unsigned int* run_indices, 
	const ModelParameters<value_type> par, 
	const value_type jac_step,
	const value_type gamma,
	value_type* jac)
{
	// determining the equation number
	unsigned int eid = threadIdx.x;

	// determining the variable number and correct shift index (ONLY one)
	unsigned int vid = 0;
	unsigned int u_shift_index = 0xffffffff, phi_shift_index = 0xffffffff;

	if (threadIdx.y < 5) // the differentiation is with respect to some of the first n variables (curvature part)
	{
		if (threadIdx.y < 2)
			u_shift_index = prev_index(eid, par.N, 2 - threadIdx.y);
		else if (threadIdx.y > 2)
			u_shift_index = next_index(eid, par.N, threadIdx.y - 2);
		else u_shift_index = eid;

		vid = u_shift_index;
	}
	else // the differentiation is with respect to chemical part
	{
		if (threadIdx.y < 6)
			phi_shift_index = prev_index(eid, par.N, 6 - threadIdx.y);
		else if (threadIdx.y > 6)
			phi_shift_index = next_index(eid, par.N, threadIdx.y - 6);
		else
			phi_shift_index = eid;

		vid = par.N + phi_shift_index;
	}


	const shift_handler<value_type, unsigned int> global_u(w_m + run_indices[blockIdx.x] * par.sys_size, u_shift_index, jac_step);
	const shift_handler<value_type, unsigned int> global_phi(global_u.data() + par.N, phi_shift_index, jac_step);
	const value_type lambda = *(global_phi.data() + par.N);

	const value_type* rhs_u = rhs_m + run_indices[blockIdx.x] * par.rhs_size; // NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	const value_type* rhs_phi = rhs_u + par.N;

	value_type dudt, dphidt;

	// evaluate the r.h.s.
	compute_rhs(global_u, global_phi, lambda, eid, par, dudt, dphidt);

	//if (vid == eid)
	//	printf("tid(1) = %i, tid(2) = %i, u[i] = %i, phi[i] = %i, vid = %i, dudt = %f, dphidt = %f \n", 
	//		threadIdx.x, threadIdx.y, u_shift_index, phi_shift_index, vid, dudt, dphidt);


	// fill the values of Jacobian matrix
	jac[run_indices[blockIdx.x] * par.sys_size * par.sys_size + par.sys_size * vid + eid] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] / jac_step * (dudt - rhs_u[eid]) - (vid == eid);

	if ((threadIdx.y != 0) && (threadIdx.y != 4))
		jac[run_indices[blockIdx.x] * par.sys_size * par.sys_size + par.sys_size * vid + eid + par.N] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] / jac_step * (dphidt - rhs_phi[eid]) - (vid == eid + par.N);
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
	unsigned int km = prev_index(k, par.N);
	unsigned int kp = next_index(k, par.N);

	const value_type* global_u = w_m + run_indices[blockIdx.x] * par.sys_size; // component for lagrangian multiplier is also taken into account
	const shift_handler<value_type, unsigned int> global_u_h(global_u, k, jac_step);

	value_type du = diffFDS(global_u, k, km, par.dxinv);
	value_type du_next = diffFDS(global_u, kp, k, par.dxinv);
	value_type d2u = diff2FDS(global_u, k, km, kp, par.dx2inv);

	const value_type t2inv = value_type(1) / (1 + du * du);
	const value_type t3inv = t2inv * sqrt(t2inv);

	value_type du_h = diffFDS(global_u_h, k, km, par.dxinv);
	value_type du_next_h = diffFDS(global_u_h, kp, k, par.dxinv);

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

// this function evaluates jacobian of the model
template <typename value_type> void evaluate_jacobian(
	const cudaStream_t t_cuda_stream,
	const value_type* t_device_w_m,
	const value_type* t_device_rhs_m,
	const value_type* t_device_arc_m,
	const value_type* t_device_dt,
	const unsigned int t_sys_size,
	const ModelParameters<value_type> t_par,
	const value_type t_jac_step,
	const unsigned int* t_device_run_indices,
	const unsigned int t_num_run,
	const value_type t_gamma,
	value_type* t_device_Jac
)
{
	{
		dim3 grid{ t_num_run,1,1 };
		dim3 block{t_par.N,8,1};
		
		evaluateJac_p1<<<grid,block,0,t_cuda_stream>>>(
			t_device_w_m,
			t_device_rhs_m,
			t_device_dt,
			t_device_run_indices,
			t_par,
			t_jac_step,
			t_gamma,
			t_device_Jac);
			
	}
	{
		dim3 grid{ t_num_run,1,1 };
		dim3 block{ t_par.N,1,1 };
		evaluateJac_p2<<<grid,block,0,t_cuda_stream>>>(
			t_device_w_m,
			t_device_dt,
			t_device_run_indices,
			t_par,
			t_jac_step,
			t_gamma,
			t_device_Jac
		);
	}
}