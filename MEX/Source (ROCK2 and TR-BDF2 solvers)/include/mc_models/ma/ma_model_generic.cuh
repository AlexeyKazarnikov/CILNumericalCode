#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mc_models/shared_routines.cuh"
#include "mc_models/shift_handler.cuh"
#include "utils/fds_utils_generic.h"

template <typename value_type> struct ModelParameters
{
public:
	const value_type beta;
	const value_type L;
	const value_type tau;
	const value_type Da;
	const value_type Di;
	const value_type mua;
	const value_type mui;
	const unsigned int N;
	const value_type dx;
	const value_type dxinv;
	const value_type dx2inv;
	const value_type dx3inv;
	const value_type dx4inv;
	const value_type scale_const;
	const unsigned int sys_dim = 3;
	const unsigned int sys_size;
	const unsigned int rhs_size;

	ModelParameters(
		value_type t_beta,
		value_type t_L,
		value_type t_tau,
		value_type t_Da,
		value_type t_Di,
		value_type t_mua,
		value_type t_mui,
		unsigned int t_N)
		:
		beta(t_beta),
		L(t_L),
		tau(t_tau),
		Da(t_Da),
		Di(t_Di),
		mua(t_mua),
		mui(t_mui),
		N(t_N),
		dx(this->L / this->N),
		dxinv(this->N / this->L),
		dx2inv(dxinv* dxinv),
		dx3inv(dx2inv* dxinv),
		dx4inv(dx2inv* dx2inv),
		scale_const(-this->L / this->tau),
		sys_size(3 * N + 1),
		rhs_size(3 * N)
	{}
};

template <typename handler_type, typename value_type> __device__ void compute_rhs(
	const handler_type& t_global_u,
	const handler_type& t_global_phi,
	const handler_type& t_global_theta,
	const value_type t_lambda,
	const unsigned int t_k,
	const ModelParameters<value_type>& t_par,
	value_type& t_dudt,
	value_type& t_dphidt,
	value_type& t_dthetadt)
{
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

	value_type phi_k = t_global_phi[t_k];
	value_type phi_k_m = get_prev_element(t_global_phi, t_k, t_par.N);
	value_type phi_k_p = get_next_element(t_global_phi, t_k, t_par.N);

	value_type theta_k = t_global_theta[t_k];
	value_type theta_k_m = get_prev_element(t_global_theta, t_k, t_par.N);
	value_type theta_k_p = get_next_element(t_global_theta, t_k, t_par.N);

	value_type u_k = t_global_u[t_k];
	value_type u_k_m = get_prev_element(t_global_u, t_k, t_par.N);
	value_type u_k_p = get_next_element(t_global_u, t_k, t_par.N);
	value_type u_k_pp = get_after_next_element(t_global_u, t_k, t_par.N);
	value_type u_k_mm = get_after_prev_element(t_global_u, t_k, t_par.N);

	value_type du = diffFDS(u_k, u_k_m, t_par.dxinv);
	value_type d2u = diff2FDS(u_k, u_k_p, u_k_m, t_par.dx2inv);
	value_type d3u = diff3FDS(u_k, u_k_p, u_k_m, u_k_pp, u_k_mm, t_par.dx3inv);
	value_type d4u = diff4FDS(u_k, u_k_p, u_k_m, u_k_pp, u_k_mm, t_par.dx4inv);

	value_type dphi = diffFDS(phi_k, phi_k_m, t_par.dxinv);
	value_type d2phi = diff2FDS(phi_k, phi_k_p, phi_k_m, t_par.dx2inv);

	value_type dtheta = diffFDS(theta_k, theta_k_m, t_par.dxinv);
	value_type d2theta = diff2FDS(theta_k, theta_k_p, theta_k_m, t_par.dx2inv);

	t2inv = value_type(1) / (1 + du * du);
	t3inv = t2inv * sqrt(t2inv);
	t4inv = t2inv * t2inv;

	du_d2u = du * d2u;
	du_d2u_t4inv = du_d2u * t4inv;

	k0 = d2u * t3inv;
	dk0 = t3inv * (d3u - 3 * du_d2u * d2u * t2inv);
	d2k0 = t3inv * (d4u - d2u * (3 * t2inv * (d2u * d2u + 3 * du * d3u) - 15 * du_d2u * du_d2u_t4inv));

	k_hat = k0 + t_par.beta * phi_k;
	dk_hat = dk0 + t_par.beta * dphi;
	d2k_hat = d2k0 + t_par.beta * d2phi;

	// evaluate the r.h.s.
	t_dudt = t_par.scale_const * (d2k_hat * t2inv - dk_hat * du_d2u_t4inv - k_hat * k0 * k0) + t_lambda * k0;
	t_dphidt = t_par.Da * (d2phi * t2inv - dphi * du_d2u_t4inv) - t_par.mua * phi_k + phi_k * phi_k / theta_k;
	t_dthetadt = t_par.Di * (d2theta * t2inv - dtheta * du_d2u_t4inv) - t_par.mui * theta_k + phi_k * phi_k;
}

template <typename value_type> __global__ void model_rhs(
	const value_type* t_w, 
	value_type* t_dwdt, 
	const unsigned int* t_run_indices, 
	const ModelParameters<value_type> t_par
)
{
	// first we determine the current system position in the global memory block
	// note that each thread corresponts to two (!) nodes of the model (surface u(k) and morphogen phi(k)).
	// that is why one block corresponds to 2 * par.Ndim elements in memory
	const value_type* global_u = t_w + t_run_indices[blockIdx.x] * t_par.sys_size; // component for lagrangian multiplier is also taken into account
	const value_type* global_phi = global_u + t_par.N;
	const value_type* global_theta = global_phi + t_par.N;
	const value_type lambda = *(global_theta + t_par.N);

	value_type* global_dudt = t_dwdt + t_run_indices[blockIdx.x] * t_par.rhs_size; // NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	value_type* global_dphidt = global_dudt + t_par.N;
	value_type* global_dthetadt = global_dphidt + t_par.N;

	value_type dudt, dphidt, dthetadt;

	// evaluate the r.h.s.
	compute_rhs(global_u, global_phi, global_theta, lambda, threadIdx.x, t_par, dudt, dphidt, dthetadt);

	global_dudt[threadIdx.x] = dudt;
	global_dphidt[threadIdx.x] = dphidt;
	global_dthetadt[threadIdx.x] = dthetadt;
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

template <typename value_type> __global__ void model_rhs_sum_norm(
	const value_type* w,
	const size_t N,
	value_type* norm,
	const unsigned int sys_dim,
	const unsigned int dim_size,
	const unsigned int* run_indices,
	const value_type dx)
{
	static __shared__ value_type shared_mem[32];

	const value_type* global_w = w + run_indices[blockIdx.x] * sys_dim * dim_size + dim_size;

	// indexes for the current thread
	const unsigned int k = threadIdx.x;
	const unsigned int lane = k % warpSize;
	const unsigned int wid = k / warpSize;

	value_type val = 0;
	if (k < N)
	{
		auto phi_k = global_w[k];
		auto theta_k = global_w[k + dim_size];
		val = phi_k * phi_k + theta_k * theta_k;
	}

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
		norm[run_indices[blockIdx.x]] = sqrt(dx * val);
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
	model_rhs_sum_norm << <grid, block, 0, t_cuda_stream >> > (t_device_rhs_m, t_par.N, t_device_norm_m, t_par.sys_dim, t_par.N, t_device_run_indices, t_par.dx);
}

template <typename value_type> __global__ void evaluateJac_p1(
	const value_type* w_m, 
	const value_type* rhs_m, 
	const value_type* dt, 
	const unsigned int* run_indices, 
	const ModelParameters<value_type> par, 
	const value_type jac_step, 
	const value_type gamma,
	value_type* jac
)
{
	// determining the equation number
	unsigned int eid = threadIdx.x;

	// determining the variable number and correct shift index (ONLY one)
	unsigned int vid = 0;
	unsigned int u_shift_index = 0xffffffff, phi_shift_index = 0xffffffff, theta_shift_index = 0xffffffff;

	if (threadIdx.y < 5) // the differentiation is with respect to some of the first n variables (curvature part)
	{
		if (threadIdx.y < 2)
		{
			if ((u_shift_index = prev_index(eid, par.N, 2 - threadIdx.y)) == prev_index(eid, par.N, 1 - threadIdx.y))
				return;
		}
		else if (threadIdx.y > 2)
		{
			if ((u_shift_index = next_index(eid, par.N, threadIdx.y - 2)) == next_index(eid, par.N, threadIdx.y - 3))
				return;
		}
		else
			u_shift_index = eid;

		vid = u_shift_index;
	}
	else if (threadIdx.y < 8) // the differentiation is with respect to chemical part (activator)
	{
		if (threadIdx.y < 6)
		{
			if ((phi_shift_index = prev_index(eid, par.N, 6 - threadIdx.y)) == eid)
				return;
		}
		else if (threadIdx.y > 6)
		{
			if ((phi_shift_index = next_index(eid, par.N, threadIdx.y - 6)) == eid)
				return;
		}
		else
			phi_shift_index = eid;

		vid = par.N + phi_shift_index;
	}
	else
	{
		if (threadIdx.y < 9)
		{
			if ((theta_shift_index = prev_index(eid, par.N, 9 - threadIdx.y)) == eid)
				return;
		}
		else if (threadIdx.y > 9)
		{
			if ((theta_shift_index = next_index(eid, par.N, threadIdx.y - 9)) == eid)
				return;
		}
		else
			theta_shift_index = eid;

		vid = 2 * par.N + theta_shift_index;
	}


	const shift_handler<value_type, unsigned int> global_u(w_m + run_indices[blockIdx.x] * par.sys_size, u_shift_index, jac_step);
	const shift_handler<value_type, unsigned int> global_phi(global_u.data() + par.N, phi_shift_index, jac_step);
	const shift_handler<value_type, unsigned int> global_theta(global_phi.data() + par.N, theta_shift_index, jac_step);
	const value_type lambda = *(global_theta.data() + par.N);

	const value_type* rhs_u = rhs_m + run_indices[blockIdx.x] * par.rhs_size; // NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	const value_type* rhs_phi = rhs_u + par.N;
	const value_type* rhs_theta = rhs_phi + par.N;

	value_type dudt, dphidt, dthetadt;

	// evaluate the r.h.s.
	compute_rhs(global_u, global_phi, global_theta, lambda, eid, par, dudt, dphidt, dthetadt);

	//if (vid == eid)
	//	printf("tid(1) = %i, tid(2) = %i, u[i] = %i, phi[i] = %i, vid = %i, dudt = %f, dphidt = %f \n", 
	//		threadIdx.x, threadIdx.y, u_shift_index, phi_shift_index, vid, dudt, dphidt);


	// fill the values of Jacobian matrix
	jac[run_indices[blockIdx.x] * par.sys_size * par.sys_size + par.sys_size * vid + eid] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] / jac_step * (dudt - rhs_u[eid]) - (vid == eid);

	if ((threadIdx.y != 0) && (threadIdx.y != 4))
		jac[run_indices[blockIdx.x] * par.sys_size * par.sys_size + par.sys_size * vid + eid + par.N] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] / jac_step * (dphidt - rhs_phi[eid]) - (vid == eid + par.N);

	jac[run_indices[blockIdx.x] * par.sys_size * par.sys_size + par.sys_size * vid + eid + 2 * par.N] =
		(gamma / value_type(2)) * dt[run_indices[blockIdx.x]] / jac_step * (dthetadt - rhs_theta[eid]) - (vid == eid + 2 * par.N);
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
		dim3 block{t_par.N,11,1};
		
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