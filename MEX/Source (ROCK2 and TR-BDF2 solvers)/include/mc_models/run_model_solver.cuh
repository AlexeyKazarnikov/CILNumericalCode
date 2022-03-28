/*
* This header file contains generic implementation of the parallel CUDA-based 
* algorithm for solving mechano-chemical model with global arc length constraint.
* Model-specific routines come at the beginning of the file and should be implemented
* for any concrete model to be used by the solver. Infinite-dimensional system should be
* discretized by a finite difference scheme. Numerical integration is done by using the
* implicit method TR-BDF2. The solution of linear systems is done with the help of NVIDIA CUBLAS
* library. The solution of non-linear system is done by using the simplified Newton's method.
*/

#pragma once

//#define TIME_MEASUREMENT

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>

#include <chrono>

#include "cublas_v2.h"

#include "mc_models/shared_routines.cuh"
#include "utils/CUDAArrayLinks_v3.cuh"
#include "utils/RunResult.cuh"

#ifdef TIME_MEASUREMENT
#include "CUDAStreamLogger.cuh"
#endif

template <typename value_type> void TRACE_LINK(DeviceArrayLink<value_type>& link, const std::string& filename)
{
	std::vector<value_type> host_data(link.size());
	link.copy_to_host(host_data);
	write_vector(filename, host_data);
}

/* MODEL-SPECIFIC ROUTINES */

// This structute should contain all model parameters. To be used in the following model-specific routines only.
template <typename value_type> struct ModelParameters;

// This function evaluates global arc length of the mechano-chemical model.
template <typename value_type> void evaluate_arc_length(
	const cudaStream_t t_cuda_stream, // CUDA stream, in which computations are running
	const value_type* t_device_w_m, // pointer to the array of state vectors of the simulated systems
	const unsigned int t_sys_size, // length of one state vector
	const ModelParameters<value_type> t_par, // model parameters (same for all simulated systems)
	const unsigned int* t_device_run_indices, // pointer to the array of indexes of the simulated systems, for which global arc length should be computed
	const unsigned int t_num_run, // total number of elements in the array of indexes
	value_type* t_device_arc_m // pointer to the array, where global arc length should be stored
);

// This function evaluates r.h.s. of the mechano-chemical model. Note that each system state contains
// an additional component for the Lagrange multiplier, which is not included into the r.h.s. vector.
template <typename value_type> void evaluate_rhs(
	const cudaStream_t t_cuda_stream, // CUDA stream, in which computations are running
	const value_type* t_device_w_m, // pointer to the array of state vectors of the simulated systems
	const unsigned int t_sys_size, // length of one state vector
	const ModelParameters<value_type> t_par, // model parameters (same for all simulated systems)
	const unsigned int* t_device_run_indices, // pointer to the array of indexes of the simulated systems, for which global arc length should be computed
	const unsigned int t_num_run, // total number of elements in the array of indexes
	value_type* t_device_rhs_m // pointer to the array, where r.h.s. should be stored
);

// This function computes the norm for the r.h.s. of the mechano-chemical model.
template <typename value_type> void evaluate_rhs_norm(
	const cudaStream_t t_cuda_stream,
	const value_type* t_device_rhs_m,
	const unsigned int t_sys_size,
	const ModelParameters<value_type> t_par,
	const unsigned int* t_device_run_indices,
	const unsigned int t_num_run,
	value_type* t_device_norm_m
);

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
);

/* NUMERICAL ALGORITHM ROUTINES */

template <typename value_type> struct SolverParameters
{
	value_type initial_step = value_type(1e-5);
	value_type omega = value_type(0.25);
	value_type abs_tol = value_type(1e-3);
	value_type rel_tol = value_type(1e-3);
	value_type time_rel_error = value_type(1e-3);
	unsigned int max_iter = 50;
	value_type max_step = value_type(1);
	value_type conv_norm = value_type(0.015);
	value_type jac_step = value_type(1e-8);
	value_type max_time = value_type(300);
	unsigned int max_step_number = 300;
	unsigned int jac_update_interval = 1;
	unsigned int monitor_step_number = 100;
	unsigned int monitor_interval = 25;
	value_type monitor_rel_error = value_type(1e-3);
};

// helper functions, which are independent from model
template <typename value_type> __global__ void evaluate_F1(
	const value_type* w_prev,
	const value_type* w_m,
	const value_type* rhs_prev,
	const value_type* rhs_m,
	const value_type* arc_prev,
	const value_type* arc_m,
	const value_type* dt,
	const value_type gamma,
	const unsigned int* run_indices,
	const ModelParameters<value_type> par,
	value_type* F_m)
{
	// NOTE that for these vectors component for lagrangian multiplier is also taken into account
	const value_type* y_prev = w_prev + run_indices[blockIdx.x] * (blockDim.x + 1);
	const value_type* y_m = w_m + run_indices[blockIdx.x] * (blockDim.x + 1);
	value_type* Fy_m = F_m + run_indices[blockIdx.x] * (blockDim.x + 1);

	// NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	const value_type* rhs_y_prev = rhs_prev + run_indices[blockIdx.x] * blockDim.x;
	const value_type* rhs_y_m = rhs_m + run_indices[blockIdx.x] * blockDim.x;

	const unsigned int k = threadIdx.x;

	/*
	wprev(1:end - 1) + gamma * dt / 2 *...
		(...
			evaluateRhsLagrangian(wprev(1:end - 1), wprev(end), par) + ...
			evaluateRhsLagrangian(wm(1:end - 1), wm(end), par)...
			) - wm(1:end - 1);
	*/

	Fy_m[k] = y_prev[k] + gamma * value_type(0.5) * dt[run_indices[blockIdx.x]] * (rhs_y_prev[k] + rhs_y_m[k]) - y_m[k];

	if (k == 0)
		Fy_m[blockDim.x] = arc_m[run_indices[blockIdx.x]] - arc_prev[run_indices[blockIdx.x]];
}

template <typename value_type> __global__ void update_solution(
	const value_type* w_m,
	const value_type* dw,
	const unsigned int* run_indices,
	const value_type omega,
	value_type* w_new)
{
	auto idx = blockDim.x * run_indices[blockIdx.x] + threadIdx.x;
	w_new[idx] = w_m[idx] - omega * dw[idx];
}

template <typename value_type> __global__ void evaluate_F2(
	const value_type* w_prev,
	const value_type* w_int,
	const value_type* w_m,
	const value_type* rhs_m,
	const value_type* arc_prev,
	const value_type* arc_m,
	const value_type* dt,
	const value_type gamma,
	const unsigned int* run_indices,
	const ModelParameters<value_type> par,
	value_type* F_m)
{
	// NOTE that for these vectors component for lagrangian multiplier is also taken into account
	const value_type* y_prev = w_prev + run_indices[blockIdx.x] * (blockDim.x + 1);
	const value_type* y_int = w_int + run_indices[blockIdx.x] * (blockDim.x + 1);
	const value_type* y_m = w_m + run_indices[blockIdx.x] * (blockDim.x + 1);
	value_type* Fy_m = F_m + run_indices[blockIdx.x] * (blockDim.x + 1);

	// NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	const value_type* rhs_y_m = rhs_m + run_indices[blockIdx.x] * blockDim.x;

	const unsigned int k = threadIdx.x;

	value_type alpha0 = (1 - gamma) / (gamma * dt[run_indices[blockIdx.x]]);
	value_type alpha1 = 1 / (gamma * (1 - gamma) * dt[run_indices[blockIdx.x]]);
	value_type alpha2 = (2 - gamma) / ((1 - gamma) * dt[run_indices[blockIdx.x]]);

	/*
	alpha1/alpha2*wint(1:end-1) - alpha0/alpha2*wprev(1:end-1) + ...
        (1/alpha2)*evaluateRhsLagrangian(wm(1:end-1),wm(end),par) - ...
        wm(1:end-1);
	*/

	Fy_m[k] = alpha1 / alpha2 * y_int[k] - alpha0 / alpha2 * y_prev[k] 
		+ (1 / alpha2) * rhs_y_m[k] - y_m[k];

	if (k == 0)
		Fy_m[blockDim.x] = arc_m[run_indices[blockIdx.x]] - arc_prev[run_indices[blockIdx.x]];
}


template <typename value_type> __global__ void compute_tau_rel(
	const value_type* w_prev, 
	const value_type* w_int,
	const value_type* w_next,
	const value_type* rhs_prev,
	const unsigned int* run_indices,
	const value_type gamma,
	const value_type* device_dt,
	value_type* tau_rel)
{
	const value_type* y_prev = w_prev + run_indices[blockIdx.x] * (blockDim.x + 1);
	const value_type* y_int = w_int + run_indices[blockIdx.x] * (blockDim.x + 1);
	const value_type* y_next = w_next + run_indices[blockIdx.x] * (blockDim.x + 1);

	// NOTE THAT THE LAGRANGIAN MULTIPLIER IS NOT PRESENTED IN THE RHS!
	const value_type* rhs = rhs_prev + run_indices[blockIdx.x] * blockDim.x;
	value_type* result = tau_rel + run_indices[blockIdx.x] * blockDim.x;

	const unsigned int k = threadIdx.x;

	value_type tau = (3 * gamma * gamma - 4 * gamma + 2) / (6 * (1 - gamma) * (1 - gamma)) *
		(y_next[k]
			- 1 / (gamma * gamma) * y_int[k] 
			+ (1 - gamma * gamma) / (gamma * gamma) * y_prev[k]
			+ device_dt[run_indices[blockIdx.x]] * (1 - gamma) / gamma * rhs[k]);

	result[k] = abs(tau  / y_next[k]);
}

template <typename value_type> inline void compute_F1(
	cudaStream_t t_cuda_stream,
	DeviceArrayLink<value_type>& t_device_w_prev,
	DeviceArrayLink<value_type>& t_device_rhs_prev,
	DeviceArrayLink<value_type>& t_device_arc_prev,
	DeviceArrayLink<value_type>& t_device_w_m,
	DeviceArrayLink<value_type>& t_device_rhs_m,
	DeviceArrayLink<value_type>& t_device_arc_m,
	DeviceArrayLink<value_type>& t_device_F_m,
	const unsigned int t_sys_size,
	ModelParameters<value_type> t_par,
	DeviceArrayLink<value_type>& t_device_dt,
	const value_type t_gamma,
	DeviceArrayLink<unsigned int>& t_device_run_indices,
	const unsigned int t_num_run
)
{
	evaluate_rhs(
		t_cuda_stream,
		t_device_w_m.data(),
		t_sys_size,
		t_par,
		t_device_run_indices.data(),
		t_num_run,
		t_device_rhs_m.data()
	);
	CUDA_LOG(cudaGetLastError());

	// arc_m_1 <- evaluate_arc(w_m_1)
	evaluate_arc_length(
		t_cuda_stream,
		t_device_w_m.data(),
		t_sys_size,
		t_par,
		t_device_run_indices.data(),
		t_num_run,
		t_device_arc_m.data()
	);
	CUDA_LOG(cudaGetLastError());

	// F_m <- F1(w_prev,w_m_1)
	{
		dim3 grid{ t_num_run, 1, 1 };
		dim3 block{ static_cast<unsigned int>(t_sys_size - 1), 1, 1 };
		evaluate_F1 << <grid, block, 0, t_cuda_stream >> > (
			t_device_w_prev.data(),
			t_device_w_m.data(),
			t_device_rhs_prev.data(),
			t_device_rhs_m.data(),
			t_device_arc_prev.data(),
			t_device_arc_m.data(),
			t_device_dt.data(),
			t_gamma,
			t_device_run_indices.data(),
			t_par,
			t_device_F_m.data()
			);
	}
}

template <typename value_type> inline void compute_F2(
	cudaStream_t t_cuda_stream,
	DeviceArrayLink<value_type>& t_device_w_prev,
	DeviceArrayLink<value_type>& t_device_arc_prev,
	DeviceArrayLink<value_type>& t_device_w_int,
	DeviceArrayLink<value_type>& t_device_w_m,
	DeviceArrayLink<value_type>& t_device_rhs_m,
	DeviceArrayLink<value_type>& t_device_arc_m,
	DeviceArrayLink<value_type>& t_device_F_m,
	const unsigned int t_sys_size,
	ModelParameters<value_type> t_par,
	DeviceArrayLink<value_type>& t_device_dt,
	const value_type t_gamma,
	DeviceArrayLink<unsigned int>& t_device_run_indices,
	const unsigned int t_num_run
)
{
	// rhs_m_2 <- evaluate_rhs(w_m_2)
	evaluate_rhs(
		t_cuda_stream,
		t_device_w_m.data(),
		t_sys_size,
		t_par,
		t_device_run_indices.data(),
		t_num_run,
		t_device_rhs_m.data()
	);
	CUDA_LOG(cudaGetLastError());

	// arc_m_2 <- evaluate_arc(w_m_2)
	evaluate_arc_length(
		t_cuda_stream,
		t_device_w_m.data(),
		t_sys_size,
		t_par,
		t_device_run_indices.data(),
		t_num_run,
		t_device_arc_m.data()
	);
	CUDA_LOG(cudaGetLastError());

	// F_m <- F2(w_prev,w_m_1,w_m_2)
	{
		dim3 grid{ t_num_run, 1, 1 };
		dim3 block{ static_cast<unsigned int>(t_sys_size - 1), 1, 1 };
		evaluate_F2 << <grid, block, 0, t_cuda_stream >> > (
			t_device_w_prev.data(),
			t_device_w_int.data(),
			t_device_w_m.data(),
			t_device_rhs_m.data(),
			t_device_arc_prev.data(),
			t_device_arc_m.data(),
			t_device_dt.data(),
			t_gamma,
			t_device_run_indices.data(),
			t_par,
			t_device_F_m.data()
			);
	}
	CUDA_LOG(cudaGetLastError());
}

template <typename value_type> inline void calculate_L2_norm(
	cudaStream_t t_cuda_stream,
	DeviceArrayLink<value_type>& t_device_w,
	DeviceArrayLink<value_type>& t_device_norm_w,
	const unsigned int t_w_size,
	DeviceArrayLink<unsigned int>& t_device_run_indices,
	const unsigned int t_num_run
)
{
	unsigned int warp_size = 32;
	unsigned int warp_count = t_w_size / warp_size + (t_w_size % warp_size != 0);
	dim3 grid{ t_num_run, 1, 1 };
	dim3 block{ warp_size * warp_count, 1, 1 };
	compute_norm << <grid, block, 0, t_cuda_stream >> > (
		t_device_w.data(),
		t_w_size,
		t_device_norm_w.data(),
		t_device_run_indices.data()
		);
}

template <typename value_type> __global__ void copy_system_state(
	const value_type* t_device_w1,
	value_type* t_device_w2,
	const unsigned int* t_device_run_indices
)
{
	auto idx = t_device_run_indices[blockIdx.x] * blockDim.x + threadIdx.x;

	t_device_w2[idx] = t_device_w1[idx];
}

__global__ void save_actual_pivot_data(
	const int* device_actual_pivot_data,
	int* device_pivot_data,
	const unsigned int* device_run_indices
)
{
	auto idx_source = blockIdx.x * blockDim.x + threadIdx.x;
	auto idx_destination = device_run_indices[blockIdx.x] * blockDim.x + threadIdx.x;

	device_pivot_data[idx_destination] = device_actual_pivot_data[idx_source];
}

__global__ void load_actual_pivot_data(
	const int* device_pivot_data,
	int* device_actual_pivot_data,
	const unsigned int* device_run_indices
)
{
	auto idx_source = device_run_indices[blockIdx.x] * blockDim.x + threadIdx.x;
	auto idx_destination = blockIdx.x * blockDim.x + threadIdx.x;

	device_actual_pivot_data[idx_destination] = device_pivot_data[idx_source];
}



template <typename value_type> void run_model_solver_thread(
	value_type* t_initial_data,
	value_type* t_output_data,
	unsigned int t_sys_size,
	unsigned int t_num_sim,
	unsigned int t_device_number,
	ModelParameters<value_type> t_par,
	SolverParameters<value_type> t_sol_par,
	RunResult& t_result)
{
	t_result = run_model_solver(t_initial_data, t_output_data, t_sys_size, t_num_sim, t_device_number, t_par, t_sol_par);
}

template <typename value_type> RunResult run_model_solver(
	value_type* t_initial_data, // vector of initial conditions
	value_type* t_output_data, // vector, where output patterns will be saved
	unsigned int t_sys_size, // dimension of the system
	unsigned int t_num_sim, // number of simulations
	unsigned int t_device_number, // number of CUDA device, where simulations will be executed
	ModelParameters<value_type> t_par, // model-specific struct with parameter values
	SolverParameters<value_type> t_sol_par // solver settings
)
{
#ifdef TIME_MEASUREMENT
	long long duration_main = 0;
	long long duration_linear = 0;
	long long duration_F1 = 0;
	long long duration_F2 = 0;
	long long duration_Jac = 0;
	long long duration_norm_F1 = 0;
	long long duration_norm_F2 = 0;
	long long duration_iter_begin = 0;
	long long duration_data_transfer = 0;
	long long duration_div_control_1 = 0;
	long long duration_div_control_2 = 0;
	long long duration_step_control = 0;
	long long duration_index_formation = 0;
	long long duration_begin = 0;
	long long duration_end = 0;
	long long duration_check = 0;
	long long duration_tau_rel = 0;
	long long duration_host_memcpy = 0;
#endif

#ifdef TIME_MEASUREMENT
	auto begin_start = std::chrono::high_resolution_clock::now();
#endif


	auto rhs_size = t_sys_size - 1;

	cudaStream_t cuda_stream;
	cublasHandle_t cublas_handle;

	// all computations in this function are performed in the specified device, so we set it at the very beginning
	CUDA_LOG(cudaSetDevice(t_device_number));

	// we try to adjuct cache settins
	//CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared), "cache setting error");

	// next we create a stream for executing the operations
	CUDA_LOG(cudaStreamCreate(&cuda_stream));

	// next we create a CUBLAS handler and associate it with the current stream
	CUDA_LOG(cublasCreate(&cublas_handle));
	CUDA_LOG(cublasSetStream(cublas_handle, cuda_stream));

	// vectors for error control
	std::vector<value_type> norm_F_m(t_num_sim);
	std::vector<value_type> norm_F_new(t_num_sim);
	std::vector<value_type> norm_w_rel(t_num_sim);
	std::vector<value_type> norm_tau_rel(t_num_sim);
	std::vector<value_type> norm_rhs(t_num_sim);

	// host links for error control vectors
	HostArrayLink<value_type> norm_F_m_link(norm_F_m);
	HostArrayLink<value_type> norm_F_new_link(norm_F_new);
	HostArrayLink<value_type> norm_w_rel_link(norm_w_rel);
	HostArrayLink<value_type> norm_tau_rel_link(norm_tau_rel);
	HostArrayLink<value_type> norm_rhs_link(norm_rhs);

	CUDA_LOG(norm_F_m_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(norm_F_new_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(norm_w_rel_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(norm_tau_rel_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(norm_rhs_link.cuda_host_register(cudaHostRegisterPortable));

	DeviceArrayLink<value_type> device_Jac(t_num_sim * t_sys_size * t_sys_size, t_device_number);

	DeviceArrayLink<value_type> device_w_prev(t_num_sim * t_sys_size, t_device_number);
	DeviceArrayLink<value_type> device_w_m_1(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_w_m_2(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_w_new_1(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_w_new_2(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_F_m(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_dw(device_w_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_w_ref(device_w_prev.size(), t_device_number);

	DeviceArrayLink<value_type> device_rhs_prev(t_num_sim * rhs_size, t_device_number);
	DeviceArrayLink<value_type> device_rhs_m_1(device_rhs_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_rhs_m_2(device_rhs_prev.size(), t_device_number);
	DeviceArrayLink<value_type> device_tau_rel(device_rhs_prev.size(), t_device_number);

	DeviceArrayLink<value_type> device_arc_m_1(t_num_sim, t_device_number);
	DeviceArrayLink<value_type> device_arc_m_2(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_arc_prev(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_F_m(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_F_new(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_w_rel(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_tau_rel(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_rhs(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_w_ref(device_arc_m_1.size(), t_device_number);
	DeviceArrayLink<value_type> device_norm_w_new(device_arc_m_1.size(), t_device_number);

	DeviceArrayLink<int> device_pivot_data(t_sys_size * t_num_sim, t_device_number);
	DeviceArrayLink<int> device_actual_pivot_data(device_pivot_data.size(), t_device_number);
	DeviceArrayLink<int> device_info_data(device_pivot_data.size(), t_device_number);

	DeviceArrayLink<value_type*> device_array_data(t_num_sim, t_device_number);
	DeviceArrayLink<value_type*> device_rhs_array_data(device_array_data.size(), t_device_number);
	DeviceArrayLink<value_type*> device_sol_array_data(device_array_data.size(), t_device_number);



	// transferring initial data from host to device
	{
		HostArrayLink<value_type> initial_data_link(t_initial_data, t_num_sim * t_sys_size);

		CUDA_LOG(initial_data_link.cuda_host_register(cudaHostRegisterPortable));

		CUDA_LOG(device_w_prev.copy_from_host_async(initial_data_link, cuda_stream));

		CUDA_LOG(initial_data_link.cuda_host_unregister());
	}



	// initializing the array data
	std::vector<value_type*> array_data(t_num_sim);
	std::vector<value_type*> rhs_array_data(t_num_sim);
	std::vector<value_type*> sol_array_data(t_num_sim);

	HostArrayLink<value_type*> array_data_link(array_data);
	HostArrayLink<value_type*> rhs_array_data_link(rhs_array_data);
	HostArrayLink<value_type*> sol_array_data_link(sol_array_data);

	CUDA_LOG(array_data_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(rhs_array_data_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(sol_array_data_link.cuda_host_register(cudaHostRegisterPortable));	

	// TR-BDF2 method implementation
	value_type gamma = value_type(2) - sqrt(value_type(2));

	std::vector <value_type> t(t_num_sim, 0);
	std::vector <unsigned int> iter(t_num_sim, 0);
	std::vector <unsigned int> step(t_num_sim, 0);
	std::vector <bool> is_active(t_num_sim, true);
	std::vector <bool> is_rejected(t_num_sim, false);
	std::vector <bool> is_first_phase_finished(t_num_sim, false);
	std::vector <unsigned int> run_indices(t_num_sim);
	std::vector<value_type> dt(t_num_sim, t_sol_par.initial_step);

	HostArrayLink<unsigned int> run_indices_link(run_indices);
	HostArrayLink<value_type> dt_link(dt);

	CUDA_LOG(run_indices_link.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(dt_link.cuda_host_register(cudaHostRegisterPortable));

	DeviceArrayLink<unsigned int> device_run_indices(t_num_sim, t_device_number);
	DeviceArrayLink<value_type> device_dt(t_num_sim, t_device_number);


	CUDA_LOG(device_dt.copy_from_host_async(dt_link, cuda_stream));

	CUDA_LOG(device_w_prev.copy_to_device_async(device_w_m_1, cuda_stream));
	CUDA_LOG(device_w_prev.copy_to_device_async(device_w_m_2, cuda_stream));

	CUDA_LOG(device_w_prev.copy_to_device_async(device_w_ref, cuda_stream));

#ifdef TIME_MEASUREMENT
	CUDA_LOG(cudaStreamSynchronize(cuda_stream));
	auto begin_end = std::chrono::high_resolution_clock::now();
	duration_begin = std::chrono::duration_cast<std::chrono::milliseconds>(begin_end - begin_start).count();
#endif

	/*
	{
		unsigned int num_run = t_num_sim;
		for (unsigned int k = 0; k < num_run; ++k)
			run_indices[k] = k;

		CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));
		CUDA_LOG(cudaStreamSynchronize(cuda_stream));

		evaluate_monitor_norm(
			cuda_stream,
			device_w_ref.data(),
			t_sys_size,
			t_par,
			device_run_indices.data(),
			num_run,
			device_norm_w_ref.data()
		);
		CUDA_LOG(cudaGetLastError());
	}
	*/

	CUDA_LOG(cudaStreamSynchronize(cuda_stream));



	{
#ifdef TIME_MEASUREMENT
		CUDAStreamLogger log_time(cuda_stream, &duration_main);
#endif

	while (true)
	{
		{
#ifdef TIME_MEASUREMENT
			CUDAStreamLogger log_time(cuda_stream, &duration_check);
#endif

			if (!std::any_of(is_active.begin(), is_active.end(), [](bool val) {return val; }))
				break;
		}

		// beginning the iteration
		//	w_m_1 = wprev;
		//	w_m_2 = wprev;
		//	is_first_phase_finished = false;
		//  is_rejected = false;
		//  arc_prev = S(w_prev)
		//  rhs_prev = rhs(w_prev)
		{
			// forming the list of involved systems [active, iter == 0] and instantly executing the update of w_m_1 and w_m_2 (separately)
			unsigned int num_run = 0;
			{
#ifdef TIME_MEASUREMENT
				CUDAStreamLogger log_time(cuda_stream, &duration_index_formation);
#endif

				for (unsigned int k = 0; k < t_num_sim; ++k)
				{
					if (!is_active[k])
						continue;

					if (iter[k] != 0)
						continue;

					run_indices[num_run] = k;
					++num_run;
				} // for
			}

			if (num_run > 0)
			{
				// uploading the list of involved systems to device
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));
					CUDA_LOG(cudaStreamSynchronize(cuda_stream));

					dim3 grid{ num_run, 1, 1 };
					dim3 block{ t_sys_size,1,1 };

					copy_system_state << <grid, block, 0, cuda_stream >> > (
						device_w_prev.data(), 
						device_w_m_1.data(), 
						device_run_indices.data());
					copy_system_state << <grid, block, 0, cuda_stream >> > (
						device_w_prev.data(),
						device_w_m_2.data(),
						device_run_indices.data());
					CUDA_LOG(cudaGetLastError());
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger(cuda_stream, &duration_iter_begin);
#endif
					// computing S(w_prev) for involved systems
					evaluate_arc_length(
						cuda_stream,
						device_w_prev.data(),
						t_sys_size,
						t_par,
						device_run_indices.data(),
						num_run,
						device_arc_prev.data()
					);
					CUDA_LOG(cudaGetLastError());

					// computing rhs(w_prev) for involved systems
					evaluate_rhs(
						cuda_stream,
						device_w_prev.data(),
						t_sys_size,
						t_par,
						device_run_indices.data(),
						num_run,
						device_rhs_prev.data()
					);
					CUDA_LOG(cudaGetLastError());
				}
			} // if
		} // begin


		// for all active systems with we check the necessity to update Jacobian matrix and invert it
		//   if mod(iter,jac_update_interval) == 0
		//     device_Jac = Jac(w_m_2);
		//     device_Jac_inv = inv(device_Jac);
		{
			unsigned int num_run = 0;
			{
#ifdef TIME_MEASUREMENT
				CUDAStreamLogger log_time(cuda_stream, &duration_index_formation);
#endif

				for (unsigned int k = 0; k < t_num_sim; ++k)
				{
					if (!is_active[k])
						continue;

					if (iter[k] % t_sol_par.jac_update_interval != 0)
						continue;

					// adding entry to data array
					array_data[num_run] = device_Jac.data() + k * t_sys_size * t_sys_size;

					// adding entry to index array
					run_indices[num_run] = k;
					++num_run;

					// cleaning the memory for the respective block in Jacobian array
					CUDA_LOG(cudaMemsetAsync(
						device_Jac.data() + k * t_sys_size * t_sys_size,
						0,
						t_sys_size * t_sys_size * sizeof(value_type),
						cuda_stream
					));
				}
			}

			if (num_run > 0)
			{
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					// uploading the list of involved systems to device
					CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));
					// uploading the data for linear systems to device
					CUDA_LOG(device_array_data.copy_from_host_async(array_data_link, 0, 0, num_run, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_Jac);
#endif

					// rhs_m <- evaluate_rhs(w_m_2)
					evaluate_rhs(
						cuda_stream,
						device_w_m_2.data(),
						t_sys_size,
						t_par,
						device_run_indices.data(),
						num_run,
						device_rhs_m_2.data()
					);
					CUDA_LOG(cudaGetLastError());

					// arc_m <- evaluate_arc(w_m_2)
					evaluate_arc_length(
						cuda_stream,
						device_w_m_2.data(),
						t_sys_size,
						t_par,
						device_run_indices.data(),
						num_run,
						device_arc_m_2.data()
					);
					CUDA_LOG(cudaGetLastError());
				
					// Jac <- evaluate_jacobian(w_m_2, rhs_m)
					evaluate_jacobian(
						cuda_stream,
						device_w_m_2.data(),
						device_rhs_m_2.data(),
						device_arc_m_2.data(),
						device_dt.data(),
						t_sys_size,
						t_par,
						t_sol_par.jac_step,
						device_run_indices.data(),
						num_run,
						gamma,
						device_Jac.data()
					);
					CUDA_LOG(cudaGetLastError());
				}

				/*
				{
					std::vector<value_type> jac_data(t_sys_size* t_sys_size* t_num_sim);
					device_Jac.copy_to_host(jac_data);
					write_vector("jac.txt", jac_data);
				}
				*/

				// CUBLAS inversion
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_Jac);
#endif

					CUDA_LOG(cublasDgetrfBatched(
						cublas_handle,
						t_sys_size,
						device_array_data.data(),
						t_sys_size,
						device_actual_pivot_data.data(),
						device_info_data.data(),
						num_run));
				}

				// saving pivoting info
				{
					dim3 grid{ num_run,1,1 };
					dim3 block{t_sys_size,1,1};
					
					save_actual_pivot_data << <grid, block, 0, cuda_stream >> > (
						device_actual_pivot_data.data(),
						device_pivot_data.data(),
						device_run_indices.data()
						);
					CUDA_LOG(cudaGetLastError());
				}
			} // if num_run > 0		

		} // inversion

		
		/*
		{
			std::vector<value_type> w_m_data(t_sys_size* t_num_sim);
			std::vector<value_type> rhs_m_data(rhs_size* t_num_sim);
			std::vector<value_type> arc_m_data(t_num_sim);

			
			std::vector<value_type> inv_jac_data(t_sys_size* t_sys_size* t_num_sim);

			device_w_m_2.copy_to_host(w_m_data);
			device_rhs_m_2.copy_to_host(rhs_m_data);
			device_arc_m_2.copy_to_host(arc_m_data);
			
			device_Jac_inv.copy_to_host(inv_jac_data);

			cudaDeviceSynchronize();

			write_vector("w_m.txt", w_m_data);
			write_vector("rhs_m.txt", rhs_m_data);
			write_vector("arc_m.txt", arc_m_data);
			
			write_vector("jac_inv.txt", inv_jac_data);
		}
		*/
		


		// phase 1
		//  Fwm1 = F1(wm1, wprev, par);
		//  dw1 = -Jmi * Fwm1;
		//  wnew1 = wm1 + omega * dw1;
		//	Fwnew1 = F1(wnew1, wprev, par);
		//  if vecnorm(Fwm1, p) <= vecnorm(Fwnew1, p)
		//	  rejectFlag = true;
		//	  break;
		//	if it > 1 && vecnorm(Fwnew1, p) < abstol && vecnorm(wm1 - wnew1, p) < reltol
		//	  finish1 = true;
		//	wm1 = wnew1;
		{
			unsigned int num_run = 0;
			{
#ifdef TIME_MEASUREMENT
				CUDAStreamLogger log_time(cuda_stream, &duration_index_formation);
#endif

				for (unsigned int k = 0; k < t_num_sim; ++k)
				{
					if (!is_active[k])
						continue;

					if (is_first_phase_finished[k])
						continue;		

					// adding entry to data array
					array_data[num_run] = device_Jac.data() + k * t_sys_size * t_sys_size;
					rhs_array_data[num_run] = device_F_m.data() + k * t_sys_size;

					// adding entry to index array
					run_indices[num_run] = k;
					++num_run;
				} // for
			}

			if (num_run > 0)
			{
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					// uploading the list of involved systems to device
					CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));
					// uploading the data for linear systems to device
					CUDA_LOG(device_array_data.copy_from_host_async(array_data_link, 0, 0, num_run, cuda_stream));
					CUDA_LOG(device_rhs_array_data.copy_from_host_async(rhs_array_data_link, 0, 0, num_run, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));

					dim3 grid{ num_run,1,1 };
					dim3 block{ t_sys_size,1,1 };

					load_actual_pivot_data << <grid, block, 0, cuda_stream >> > (
						device_pivot_data.data(),
						device_actual_pivot_data.data(),
						device_run_indices.data());
					CUDA_LOG(cudaGetLastError());
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_F1);
#endif

					compute_F1(
						cuda_stream,
						device_w_prev,
						device_rhs_prev,
						device_arc_prev,
						device_w_m_1,
						device_rhs_m_1,
						device_arc_m_1,
						device_F_m,
						t_sys_size,
						t_par,
						device_dt,
						gamma,
						device_run_indices,
						num_run
					);
				}

				// computing ||F_m||
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F1);
#endif

					
					/*
					unsigned int warp_size = 32;
					unsigned int warp_count = t_sys_size / warp_size + (t_sys_size % warp_size != 0);
					dim3 grid{ num_run, 1, 1 };
					dim3 block{ warp_size * warp_count, 1, 1 };
					compute_norm << <grid, block, 0, cuda_stream >> > (
						device_F_m.data(),
						t_sys_size,
						device_norm_F_m.data(),
						device_run_indices.data()
						);
					*/
					calculate_L2_norm(
						cuda_stream,
						device_F_m,
						device_norm_F_m,
						t_sys_size,
						device_run_indices,
						num_run);

					CUDA_LOG(cudaGetLastError());
				}
				
				/*
				{
					std::vector<value_type> F_data(device_F_m.size());
					device_F_m.copy_to_host(F_data);
					write_vector("F_data.txt", F_data);
				}
				*/

				
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_linear);
#endif

					std::vector<int> info(num_run);
					CUDA_LOG(cublasDgetrsBatched(
						cublas_handle,
						CUBLAS_OP_N,
						t_sys_size,
						1,
						device_array_data.data(),
						t_sys_size,
						device_actual_pivot_data.data(),
						device_rhs_array_data.data(),
						t_sys_size,
						info.data(),
						num_run)
					);
				}
				


				/*
				// dw_1 = inv(Jm) * F_m
				value_type alpha1 = 1;
				value_type beta1 = 0;
				CUDA_LOG(cublasDgemmBatched(cublas_handle,
					CUBLAS_OP_N,
					CUBLAS_OP_N,
					t_sys_size,
					1,
					t_sys_size,
					&alpha1,
					device_inv_array_data.data(),
					t_sys_size,
					device_rhs_array_data.data(),
					t_sys_size,
					&beta1,
					device_sol_array_data.data(),
					t_sys_size,
					num_run)
				);
				*/

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F1);
#endif

					// wnew1 = wm1 + omega * dw1
					{
						dim3 grid{ num_run, 1, 1 };
						dim3 block{ t_sys_size, 1, 1 };
						update_solution << <grid, block, 0, cuda_stream >> > (
							device_w_m_1.data(),
							device_F_m.data(),
							device_run_indices.data(),
							t_sol_par.omega,
							device_w_new_1.data()
							);
					}
				}

				/*
				{
					TRACE_LINK(device_F_m, "dw.txt");
					TRACE_LINK(device_w_new_1, "w_new.txt");
				}
				*/

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_F1);
#endif

					compute_F1(
						cuda_stream,
						device_w_prev,
						device_rhs_prev,
						device_arc_prev,
						device_w_new_1,
						device_rhs_m_1,
						device_arc_m_1,
						device_F_m,
						t_sys_size,
						t_par,
						device_dt,
						gamma,
						device_run_indices,
						num_run
					);
					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}
				
				/*
				{
					TRACE_LINK(device_F_m, "F_new.txt");
				}	
				*/

				// computing ||F_m||
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F1);
#endif

					/*
					unsigned int warp_size = 32;
					unsigned int warp_count = t_sys_size / warp_size + (t_sys_size % warp_size != 0);
					dim3 grid{ num_run, 1, 1 };
					dim3 block{ warp_size * warp_count, 1, 1 };
					compute_norm << <grid, block, 0, cuda_stream >> > (
						device_F_m.data(),
						t_sys_size,
						device_norm_F_new.data(), 
						device_run_indices.data()
						);
					*/

					calculate_L2_norm(
						cuda_stream,
						device_F_m,
						device_norm_F_new,
						t_sys_size,
						device_run_indices,
						num_run
					);
				}

				// computing ||w_new_1 - w_m_1||
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F1);
#endif

					unsigned int warp_size = 32;
					unsigned int warp_count = t_sys_size / warp_size + (t_sys_size % warp_size != 0);
					dim3 grid{ num_run, 1, 1 };
					dim3 block{ warp_size * warp_count, 1, 1 };
					compute_diff_norm << <grid, block, 0, cuda_stream >> > (
						device_w_m_1.data(),
						device_w_new_1.data(),
						t_sys_size,
						device_norm_w_rel.data(),
						device_run_indices.data()
						);

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					// copying the norm data back to host for divergence control
					CUDA_LOG(device_norm_F_m.copy_to_host_async(norm_F_m_link, cuda_stream));
					CUDA_LOG(device_norm_F_new.copy_to_host_async(norm_F_new_link, cuda_stream));
					CUDA_LOG(device_norm_w_rel.copy_to_host_async(norm_w_rel_link, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}		

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_div_control_1);
#endif

					// divergence control (first phase)
					std::fill(is_rejected.begin(), is_rejected.end(), false);

					num_run = 0;

					for (unsigned int k = 0; k < t_num_sim; ++k)
					{
						if (!is_active[k])
							continue;

						if (is_first_phase_finished[k])
							continue;

						//std::cout << "data " << norm_F_m[k] << " " << norm_F_new[k] << std::endl;

						if (norm_F_m[k] < norm_F_new[k]) // divergence detected, aborting the iterations and reducing the time step
						{
							is_rejected[k] = true;
							iter[k] = 0;
							dt[k] /= 2;
						}
						else // everything is normal, we may try to finish the first phase of the iteration process
						{
							if ((iter[k] > 0) && (norm_F_new[k] < t_sol_par.abs_tol) && (norm_w_rel[k] < t_sol_par.rel_tol))
								is_first_phase_finished[k] = true;

							run_indices[num_run] = k;
							++num_run;
						}
					} // for

					if (std::any_of(is_rejected.begin(), is_rejected.end(), [](bool val) {return val; }))
					{
						CUDA_LOG(device_dt.copy_from_host_async(dt_link, cuda_stream));
						CUDA_LOG(cudaStreamSynchronize(cuda_stream));
					}

					if (num_run > 0)
					{
						CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));

						CUDA_LOG(cudaStreamSynchronize(cuda_stream));

						{
							//CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
							dim3 grid{ num_run,1,1 };
							dim3 block{ t_sys_size,1,1 };

							/*
							CUDA_LOG(device_w_new_1.copy_to_device_async(
								device_w_m_1,
								k* t_sys_size,
								k* t_sys_size,
								t_sys_size,
								cuda_stream));
							*/

							copy_system_state << <grid, block, 0, cuda_stream >> > (device_w_new_1.data(), device_w_m_1.data(), device_run_indices.data());
							CUDA_LOG(cudaGetLastError());
						}
					}
				}
			} // if num_run > 0
		} // phase 1

		// phase 2
		{
			unsigned int num_run = 0;
			{
#ifdef TIME_MEASUREMENT
				CUDAStreamLogger log_time(cuda_stream, &duration_index_formation);
#endif

				for (unsigned int k = 0; k < t_num_sim; ++k)
				{
					if (!is_active[k])
						continue;

					if (is_rejected[k])
						continue;

					// adding entry to data array
					array_data[num_run] = device_Jac.data() + k * t_sys_size * t_sys_size;
					rhs_array_data[num_run] = device_F_m.data() + k * t_sys_size;

					// adding entry to index array
					run_indices[num_run] = k;
					++num_run;

				} // for
			}

			if (num_run > 0)
			{
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					// uploading the list of involved systems to device
					CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));

					// uploading the data for linear systems to device
					CUDA_LOG(device_array_data.copy_from_host_async(array_data_link, 0, 0, num_run, cuda_stream));
					//CUDA_LOG(device_inv_array_data.copy_from_host_async(inv_array_data_link, 0, 0, num_run, cuda_stream));
					CUDA_LOG(device_rhs_array_data.copy_from_host_async(rhs_array_data_link, 0, 0, num_run, cuda_stream));
					//CUDA_LOG(device_sol_array_data.copy_from_host_async(sol_array_data_link, 0, 0, num_run, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));

					dim3 grid{ num_run,1,1 };
					dim3 block{ t_sys_size,1,1 };

					load_actual_pivot_data << <grid, block, 0, cuda_stream >> > (
						device_pivot_data.data(),
						device_actual_pivot_data.data(),
						device_run_indices.data());
					CUDA_LOG(cudaGetLastError());
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_F2);
#endif

					compute_F2(
						cuda_stream,
						device_w_prev,
						device_arc_prev,
						device_w_m_1,
						device_w_m_2,
						device_rhs_m_2,
						device_arc_m_2,
						device_F_m,
						t_sys_size,
						t_par,
						device_dt,
						gamma,
						device_run_indices,
						num_run
					);
					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}

				// computing ||F_m||
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F2);
#endif
					/*
					unsigned int warp_size = 32;
					unsigned int warp_count = t_sys_size / warp_size + (t_sys_size % warp_size != 0);
					dim3 grid{ num_run, 1, 1 };
					dim3 block{ warp_size * warp_count, 1, 1 };
					compute_norm << <grid, block, 0, cuda_stream >> > (
						device_F_m.data(),
						t_sys_size,
						device_norm_F_m.data(),
						device_run_indices.data()
						);
					*/

					calculate_L2_norm(
						cuda_stream,
						device_F_m,
						device_norm_F_m,
						t_sys_size,
						device_run_indices,
						num_run
					);

					CUDA_LOG(cudaGetLastError());
				}
				

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_linear);
#endif

					std::vector<int> info(num_run);
					CUDA_LOG(cublasDgetrsBatched(
						cublas_handle,
						CUBLAS_OP_N,
						t_sys_size,
						1,
						device_array_data.data(),
						t_sys_size,
						device_actual_pivot_data.data(),
						device_rhs_array_data.data(),
						t_sys_size,
						info.data(),
						num_run)
					);				
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F2);
#endif

					// wnew2 = wm2 + omega * dw
					{
						dim3 grid{ num_run, 1, 1 };
						dim3 block{ t_sys_size, 1, 1 };
						update_solution << <grid, block, 0, cuda_stream >> > (
							device_w_m_2.data(),
							device_F_m.data(),
							device_run_indices.data(),
							t_sol_par.omega,
							device_w_new_2.data()
							);
					}
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_F2);
#endif

					compute_F2(
						cuda_stream,
						device_w_prev,
						device_arc_prev,
						device_w_m_1,
						device_w_new_2,
						device_rhs_m_2,
						device_arc_m_2,
						device_F_m,
						t_sys_size,
						t_par,
						device_dt,
						gamma,
						device_run_indices,
						num_run
					);
					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}

				// computing ||F_m||
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F2);
#endif

					calculate_L2_norm(
						cuda_stream,
						device_F_m,
						device_norm_F_new,
						t_sys_size,
						device_run_indices,
						num_run
					);

					CUDA_LOG(cudaGetLastError());
				}
				

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_norm_F2);
#endif

					// computing ||w_new_2 - w_m_2||
					{
						unsigned int warp_size = 32;
						unsigned int warp_count = t_sys_size / warp_size + (t_sys_size % warp_size != 0);
						dim3 grid{ num_run, 1, 1 };
						dim3 block{ static_cast<unsigned int>(warp_size) * warp_count, 1, 1 };
						compute_diff_norm << <grid, block, 0, cuda_stream >> > (
							device_w_m_2.data(),
							device_w_new_2.data(),
							t_sys_size,
							device_norm_w_rel.data(),
							device_run_indices.data()
							);
					}
					CUDA_LOG(cudaGetLastError());
					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}	

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					// copying the norm data back to host for divergence control
					CUDA_LOG(device_norm_F_m.copy_to_host_async(norm_F_m_link, cuda_stream));
					CUDA_LOG(device_norm_F_new.copy_to_host_async(norm_F_new_link, cuda_stream));
					CUDA_LOG(device_norm_w_rel.copy_to_host_async(norm_w_rel_link, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_div_control_2);
#endif

					num_run = 0;

					// divergence control (second phase)
					for (unsigned int k = 0; k < t_num_sim; ++k)
					{
						if (!is_active[k])
							continue;

						if (is_rejected[k])
							continue;

						run_indices[num_run] = k;
						++num_run;

						if ((iter[k] > 0) && is_first_phase_finished[k] && (norm_F_new[k] < t_sol_par.abs_tol) && (norm_w_rel[k] < t_sol_par.rel_tol))
						{
							is_first_phase_finished[k] = false;
							iter[k] = 0; // exiting the iteration process
						}
						else
						{
							++iter[k];

							if (iter[k] > t_sol_par.max_iter) // divergence detected, aborting the iterations and reducing the time step
							{
								is_first_phase_finished[k] = false;
								is_rejected[k] = true;
								iter[k] = 0;
								dt[k] /= 2;
							}
						}
					} // for

					if (std::any_of(is_rejected.begin(), is_rejected.end(), [](bool val) {return val; }))
					{
						CUDA_LOG(device_dt.copy_from_host_async(dt_link, cuda_stream));
						CUDA_LOG(cudaStreamSynchronize(cuda_stream));
					}


					if (num_run > 0)
					{
						CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));

						CUDA_LOG(cudaStreamSynchronize(cuda_stream));

						{
							//CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
							dim3 grid{ num_run,1,1 };
							dim3 block{ t_sys_size,1,1 };

							copy_system_state<<<grid,block,0,cuda_stream>>>(device_w_new_2.data(), device_w_m_2.data(), device_run_indices.data());
							CUDA_LOG(cudaGetLastError());
						}
					}
				}
			}
		} // phase 2

		// step and convergence control
		//tau = (3 * gamma ^ 2 - 4 * gamma + 2) / (6 * (1 - gamma) ^ 2) * ...
		//	(wnew(1:end - 1) ...
		//		- 1 / gamma ^ 2 * wint(1:end - 1) ...
		//		+ (1 - gamma ^ 2) / gamma ^ 2 * wprev(1:end - 1) + ...
		//		dt * (1 - gamma) / gamma * evaluateRhsLagrangian(wprev(1:end - 1), wprev(end), par) ...
		//		);
		//taur = abs(tau . / wnew(1:end - 1));
		//t_opt = dt * (relerr / norm(taur)) ^ (1 / 3);
		{
			unsigned int num_run = 0;
			{
#ifdef TIME_MEASUREMENT
				CUDAStreamLogger log_time(cuda_stream, &duration_index_formation);
#endif

				for (unsigned int k = 0; k < t_num_sim; ++k)
				{
					if (!is_active[k])
						continue;

					if (is_rejected[k])
						continue;

					if (iter[k] != 0)
						continue;

					// adding entry to index array
					run_indices[num_run] = k;
					++num_run;
				} // for
			}

			if (num_run > 0)
			{
				// uploading the list of involved systems to device
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}		

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_tau_rel);
#endif

					// compute tau_rel
					{
						dim3 grid{ num_run, 1, 1 };
						dim3 block{ rhs_size, 1, 1 };
						compute_tau_rel << <grid, block, 0, cuda_stream >> > (
							device_w_prev.data(),
							device_w_new_1.data(),
							device_w_new_2.data(),
							device_rhs_prev.data(),
							device_run_indices.data(),
							gamma,
							device_dt.data(),
							device_tau_rel.data());
					}
					CUDA_LOG(cudaGetLastError());

					/*
					{
						std::vector<value_type> tau_data(rhs_size);
						device_tau_rel.copy_to_host(tau_data);
						write_vector("tau_data.txt", tau_data);
					}
					*/

					// compute ||tau_rel||
					/*
					{
						unsigned int warp_size = 32;
						unsigned int warp_count = rhs_size / warp_size + (rhs_size % warp_size != 0);
						dim3 grid{ num_run, 1, 1 };
						dim3 block{ warp_size * warp_count, 1, 1 };
						compute_norm << <grid, block, 0, cuda_stream >> > (
							device_tau_rel.data(),
							rhs_size,
							device_norm_tau_rel.data(),
							device_run_indices.data()
							);
					}
					*/

					calculate_L2_norm(
						cuda_stream,
						device_tau_rel,
						device_norm_tau_rel,
						rhs_size,
						device_run_indices,
						num_run
					);

					CUDA_LOG(cudaGetLastError());

					// compute ||rhs||
					evaluate_rhs_norm(cuda_stream,
						device_rhs_m_2.data(),
						t_sys_size,
						t_par,
						device_run_indices.data(),
						num_run,
						device_norm_rhs.data()
					);
					CUDA_LOG(cudaGetLastError());

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}		

				// copying the norm data back to host for step control
				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
#endif

					CUDA_LOG(device_norm_tau_rel.copy_to_host_async(norm_tau_rel_link, cuda_stream));
					CUDA_LOG(device_norm_rhs.copy_to_host_async(norm_rhs_link, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));
				}		

				{
#ifdef TIME_MEASUREMENT
					CUDAStreamLogger log_time(cuda_stream, &duration_step_control);
#endif

					num_run = 0;

					// time step control and finishing the step
					for (unsigned int k = 0; k < t_num_sim; ++k)
					{
						if (!is_active[k])
							continue;

						if (is_rejected[k])
							continue;

						if (iter[k] != 0)
							continue;

						// convergence control!
						if (norm_rhs[k] < t_sol_par.conv_norm)
						{
							is_active[k] = false;
							continue;
						}

						if (t[k] > t_sol_par.max_time)
						{
							is_active[k] = false;
							continue;
						}

						if (step[k] > t_sol_par.max_step_number)
						{
							is_active[k] = false;
							continue;
						}

						if (std::isnan(norm_rhs[k]))
						{
							is_active[k] = false;
							continue;
						}

						run_indices[num_run] = k;
						++num_run;

						// finishing the step
						t[k] += dt[k];

						//std::cout << "t = " << t[k] << std::endl;

						// controlling the step size
						value_type dt_opt = dt[k] * std::cbrt((t_sol_par.time_rel_error / norm_tau_rel[k]));
						// t_opt = dt * (relerr / norm(taur)) ^ (1 / 3);
						dt[k] = value_type(0.9) * dt_opt;

						++step[k];
					} // for

					// before going to text iteration, copying data and updating time steps
					CUDA_LOG(device_dt.copy_from_host_async(dt_link, cuda_stream));

					if (num_run > 0)
						CUDA_LOG(device_run_indices.copy_from_host_async(run_indices_link, 0, 0, num_run, cuda_stream));

					CUDA_LOG(cudaStreamSynchronize(cuda_stream));

					if (num_run > 0)
					{
						//CUDAStreamLogger log_time(cuda_stream, &duration_data_transfer);
						dim3 grid{ num_run,1,1 };
						dim3 block{ t_sys_size,1,1 };

						copy_system_state << <grid, block, 0, cuda_stream >> > (
							device_w_new_2.data(), 
							device_w_prev.data(), 
							device_run_indices.data());
						CUDA_LOG(cudaGetLastError());
					}
				}
				
			} // if num_run > 0
		} // step finishing

	} // while
	}

	{
#ifdef TIME_MEASUREMENT
		CUDAStreamLogger log_time(cuda_stream, &duration_end);
#endif
		
		// copying the output data back to host
		{
			HostArrayLink<value_type> output_data_link(t_output_data, t_num_sim * t_sys_size);

			CUDA_LOG(output_data_link.cuda_host_register(cudaHostRegisterPortable));

			CUDA_LOG(device_w_prev.copy_to_host_async(output_data_link, cuda_stream));

			CUDA_LOG(cudaStreamSynchronize(cuda_stream));

			CUDA_LOG(output_data_link.cuda_host_unregister());
		}

		// unregistering host links
		run_indices_link.cuda_host_unregister();
		dt_link.cuda_host_unregister();

		array_data_link.cuda_host_unregister();
		rhs_array_data_link.cuda_host_unregister();
		sol_array_data_link.cuda_host_unregister();

		// unregistering error control vectors
		norm_F_m_link.cuda_host_unregister();
		norm_F_new_link.cuda_host_unregister();

		CUDA_LOG(cublasDestroy(cublas_handle));
	}

	CUDA_LOG(cudaStreamDestroy(cuda_stream));

#ifdef TIME_MEASUREMENT
	std::cout << "Duration (main cycle): " << duration_main / 1000 << std::endl;
	std::cout << "Duration (linear systems): " << duration_linear / 1000 << std::endl;
	std::cout << "Duration (F1): " << duration_F1 / 1000 << std::endl;
	std::cout << "Duration (F2): " << duration_F2 / 1000 << std::endl;
	std::cout << "Duration (norm F1): " << duration_norm_F1 / 1000 << std::endl;
	std::cout << "Duration (norm F2): " << duration_norm_F2 / 1000 << std::endl;
	std::cout << "Duration (Jac): " << duration_Jac / 1000 << std::endl;
	std::cout << "Duration (iter begin): " << duration_iter_begin / 1000 << std::endl;
	std::cout << "Duration (data transfer): " << duration_data_transfer / 1000 << std::endl;
	std::cout << "Duration (div control 1): " << duration_div_control_1 / 1000 << std::endl;
	std::cout << "Duration (div control 2): " << duration_div_control_2 / 1000 << std::endl;
	std::cout << "Duration (step control): " << duration_step_control / 1000 << std::endl;
	std::cout << "Duration (tau rel): " << duration_tau_rel / 1000 << std::endl;
	std::cout << "Duration (index formation): " << duration_index_formation / 1000 << std::endl;
	std::cout << "Duration (begin): " << duration_begin << std::endl;
	std::cout << "Duration (end): " << duration_end / 1000 << std::endl;
	std::cout << "Duration (check): " << duration_check / 1000 << std::endl;
	std::cout << "Duration (host memcpy): " << duration_host_memcpy / 1000 << std::endl;
#endif


	return RunResult(0, "");

	/*
	if ((t < t_sol_par.max_time) && (step < t_sol_par.max_step_number))
		return RunResult(0, "");
	else
		return RunResult(0, "Convergence has not been reached!");
		*/
}