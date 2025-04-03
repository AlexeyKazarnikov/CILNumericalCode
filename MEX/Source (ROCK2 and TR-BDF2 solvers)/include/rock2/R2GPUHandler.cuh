#pragma once
/*
* This class provides a base for any GPU-based implementation of the R2Handler
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "rock2/R2GPUScheduler.cuh"
#include "rock2/R2Handler.h"

#include "utils/CUDALogger.cuh"
#include "utils/RunResult.cuh"

#include "utils/reduce_utils.cuh"

//#define LOG_TIME

/* Model-independent device constants */

# define constant_type double

__constant__ unsigned short dc_ms[46] =
{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 43,
47, 51, 56, 61, 66, 72, 78, 85, 93, 102, 112, 123,
135, 148, 163, 180, 198 };

__constant__ constant_type dc_fp1[46];

__constant__ constant_type dc_fp2[46];

__constant__ constant_type dc_recf[4476];

__constant__ constant_type dc_par[50];

__constant__ constant_type dc_atol;

__constant__ constant_type dc_rtol;

/* Model-independent CUDA kernels */

template <typename value_type, typename index_type>
__global__ void prev_state_overwrite_stage(
	value_type* t_sys_state,
	const index_type* t_run_indices,
	const index_type t_sys_size,
	const index_type sim_number
)
{
	auto tid = threadIdx.x;
	auto fid = blockIdx.x * blockDim.x + tid;
	auto vid = fid / t_sys_size;
	auto gid = fid % t_sys_size;
	if (fid < t_sys_size * sim_number)
	{
		value_type* y_start = t_sys_state + 4 * t_sys_size * t_run_indices[2 * vid];
		value_type* y_prev = y_start + t_sys_size;

		auto counter = t_run_indices[2 * vid + 1];
		if (counter < 2)
			counter = 2;
		auto y_shift = 2 - (counter) % 3;
		value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * t_sys_size;

		y_prev[gid] = y[gid];
	}
}

template <typename value_type, typename index_type>
__global__ void update_time_step_data(
	const value_type* t_temp_data,
	value_type* t_time_step_data,
	const index_type* t_run_indices,
	const index_type t_sim_number
)
{
	auto mid = blockIdx.x * blockDim.x + threadIdx.x;
	if (mid < t_sim_number)
	{
		auto sid = t_run_indices[3 * mid];
		t_time_step_data[sid] = t_temp_data[mid];
	}
}

// yjm2=yjm1; yjm1 = y;
template <typename value_type, typename index_type>
__global__ void r2_rec_shift_stage(
	value_type* t_sys_state,
	const index_type* t_run_indices,
	const index_type t_sys_size,
	const index_type sim_number
)
{
	auto tid = threadIdx.x;
	auto fid = blockIdx.x * blockDim.x + tid;
	auto vid = fid / t_sys_size;
	auto gid = fid % t_sys_size;
	if (fid < t_sys_size * sim_number)
	{
		value_type* y = t_sys_state + 4 * t_sys_size * t_run_indices[vid];
		value_type* yjm1 = y + 2 * t_sys_size;
		value_type* yjm2 = yjm1 + t_sys_size;

		yjm2[gid] = yjm1[gid];
		yjm1[gid] = y[gid];
	}
}

// t_ci1 = max(abs(y),abs(yn))*rto; err = sum((temp2*(y-yjm2)./(ato+t_ci1)).^2); err=sqrt(err/neqn);
template <typename value_type, typename index_type>
__global__ void local_error_est_stage(
	const value_type* t_sys_state,
	const value_type* t_time_steps,
	value_type* t_error,
	const index_type* t_run_indices,
	const index_type t_sys_size,
	const index_type t_sim_number // number of simulations
)
{
	static __shared__ value_type shared_mem[32];

	auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
	auto mid = blockIdx.x; // index of model which is processed
	value_type val = 0;

	auto sid = t_run_indices[2 * mid];
	auto pos_fp = t_run_indices[2 * mid + 1];
	auto counter = dc_ms[pos_fp];
	if (counter < 2)
		counter = 2;

	const value_type* y_start = t_sys_state + 4 * t_sys_size * sid;
	const value_type* y_prev = y_start + t_sys_size;

	auto y_shift = 2 - (counter) % 3;
	//auto yjm1_shift = 2 - (counter - 1) % 3;
	auto yjm2_shift = 2 - (counter - 2) % 3;

	const value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * t_sys_size;
	//const value_type* yjm1 = y_start + (yjm1_shift > 0) * (yjm1_shift + 1) * t_sys_size;
	const value_type* yjm2 = y_start + (yjm2_shift > 0) * (yjm2_shift + 1) * t_sys_size;

	value_type time_step = t_time_steps[sid];

	//value_type temp2 = time_step * dc_fp2[pos_fp];

	/*
	if (tid == 0 && blockIdx.x == 0)
	{
		int id1 = 0, id2 = 1, id3 = 512;
		printf("y[%i] = %.15f, y[%i] = %.15f, y[%i] = %.15f \n", id1, y[id1], id2, y[id2], id3, y[id3]);
		printf("y_prev[%i] = %.15f, y_prev[%i] = %.15f, y_prev[%i] = %.15f \n", id1, y_prev[id1], id2, y_prev[id2], id3, y_prev[id3]);
		//printf("yjm1[%i] = %.15f, yjm1[%i] = %.15f, yjm1[%i] = %.15f \n", id1, yjm1[id1], id2, yjm1[id2], id3, yjm1[id3]);
		printf("yjm2[%i] = %.15f, yjm2[%i] = %.15f, yjm2[%i] = %.15f \n", id1, yjm2[id1], id2, yjm2[id2], id3, yjm2[id3]);
	}
	*/

	auto gid = tid;
	while (gid < t_sys_size)
	{
		// t_ci1 = max(abs(y),abs(yn))*rto; err = sum((temp2*(y-yjm2)./(ato+t_ci1)).^2);
		value_type val_u = yjm2[gid] / (dc_atol + max(abs(y[gid]), abs(y_prev[gid])) * dc_rtol);

		/*
		if (tid == 31 && blockIdx.x == 0)
		{
			printf("val[%i] = %.12f \n", gid, val_u);
		}
		*/

		val += (val_u * val_u);

		gid += blockDim.x;
	} // while

	__syncthreads();

	// reduction

	// indexes for the current thread
	const unsigned int lane = tid % warpSize;
	const unsigned int wid = tid / warpSize;

	// warp-level reduction
	val = warp_reduce_sum(val);

	// writing reduced values to shared memory
	if (lane == 0)
		shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (tid < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_sum(val); //Final reduce within first warp

	if (tid == 0)
	{
		t_error[mid] = sqrt(val / t_sys_size);
	}
}

template <typename value_type, typename index_type>
__global__ void rhs_l2_norm_stage(
	const value_type* t_rhs_state,
	value_type* t_rhs_norm,
	const index_type* t_run_indices,
	const index_type t_sys_size,
	const index_type t_sim_number // number of simulations
)
{
	static __shared__ value_type shared_mem[32];

	auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
	auto mid = blockIdx.x; // index of model which is processed
	value_type val = 0;

	auto sid = t_run_indices[mid];

	const value_type* y_rhs = t_rhs_state + t_sys_size * sid;

	auto gid = tid;
	while (gid < t_sys_size)
	{
		value_type val_u = y_rhs[gid];
		val += (val_u * val_u);
		gid += blockDim.x;
	} // while

	__syncthreads();

	// reduction

	// indexes for the current thread
	const unsigned int lane = tid % warpSize;
	const unsigned int wid = tid / warpSize;

	// warp-level reduction
	val = warp_reduce_sum(val);

	// writing reduced values to shared memory
	if (lane == 0)
		shared_mem[wid] = val;

	__syncthreads();

	//read from shared memory only if that warp existed
	val = (tid < blockDim.x / warpSize) ? shared_mem[lane] : 0;

	if (wid == 0)
		val = warp_reduce_sum(val); //Final reduce within first warp

	if (tid == 0)
	{
		t_rhs_norm[mid] = sqrt(val) / t_sys_size;
	}
}

template <typename value_type, typename index_type>
class R2GPUHandler : public R2Handler<value_type, index_type, R2GPUScheduler<value_type, index_type>>
{
private:
	cudaError_t init_constant_arrays();
	cudaError_t m_cuda_status;

protected:
	index_type m_device_number;

	unsigned int m_block_size_overwrite = 1024;
	unsigned int m_block_size_update = 1024;
	unsigned int m_block_size_local_error = 64;
	unsigned int m_block_size_rhs_norm = 64;

	index_type find_block_number(index_type t_total_size, index_type t_block_size);

	virtual void execute_initial_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) = 0;
	virtual void execute_rec_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) = 0;
	virtual void execute_fp_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) = 0;
	virtual void execute_spectral_radius_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) = 0;

public:
	R2GPUHandler(
		index_type t_sim_number,
		index_type t_sys_size,
		const std::vector<value_type>& t_model_parameters,
		index_type t_device_number,
		value_type t_atol = 1e-3,
		value_type t_rtol = 1e-3
	);

	void handle_computations(R2GPUScheduler<value_type, index_type>& scheduler) override;

	cudaError_t get_cuda_status() { return this->m_cuda_status; }

	std::vector<long long> m_times;
};

template<typename value_type, typename index_type>
cudaError_t R2GPUHandler<value_type, index_type>::init_constant_arrays()
{
	auto cuda_error = cudaSetDevice(this->m_device_number);
	if (cuda_error != cudaSuccess)
			return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_fp1, this->m_fp1.data(), this->m_fp1.size() * sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_fp2, this->m_fp2.data(), this->m_fp2.size() * sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_recf, this->m_recf.data(), this->m_recf.size() * sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_par, this->m_model_parameters.data(), this->m_model_parameters.size() * sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_atol, &this->m_atol, sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	cuda_error = cudaMemcpyToSymbol(dc_rtol, &this->m_rtol, sizeof(constant_type));
	if (cuda_error != cudaSuccess)
		return cuda_error;

	return cuda_error;
}

template<typename value_type, typename index_type>
R2GPUHandler<value_type, index_type>::R2GPUHandler(
	index_type t_sim_number,
	index_type t_sys_size,
	const std::vector<value_type>& t_model_parameters,
	index_type t_device_number,
	value_type t_atol,
	value_type t_rtol
)
	: R2Handler<value_type, index_type, R2GPUScheduler<value_type, index_type>>(t_sim_number, t_sys_size, t_model_parameters, t_atol, t_rtol),
	m_device_number(t_device_number)
{
	CUDA_LOG(this->init_constant_arrays());
	this->m_times.resize(7);
}

template<typename value_type, typename index_type>
index_type R2GPUHandler<value_type, index_type>::find_block_number(index_type t_total_size, index_type t_block_size)
{
	return t_total_size / t_block_size
		+ (t_total_size % t_block_size != 0);
}

template<typename value_type, typename index_type>
void R2GPUHandler<value_type, index_type>::handle_computations(R2GPUScheduler<value_type, index_type>& scheduler)
{
	index_type queue_length = scheduler.get_queue_length(R2OperationType::PreviousStepOverwrite);
	size_t run_indices_offset = scheduler.get_queue_start(R2OperationType::PreviousStepOverwrite);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[0], scheduler.get_cuda_stream());
#endif

		unsigned int block_number = find_block_number(
			static_cast<unsigned int>(this->m_sys_size * queue_length),
			this->m_block_size_overwrite
		);

		dim3 grid{ block_number, 1, 1 };
		dim3 block{ this->m_block_size_overwrite, 1, 1 };
		prev_state_overwrite_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_sys_size,
			queue_length);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::InitialStage);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::InitialStage);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[1], scheduler.get_cuda_stream());
#endif
		
		unsigned int block_number = find_block_number(
			static_cast<unsigned int>(queue_length), 
			this->m_block_size_update
		);

		dim3 grid{ block_number, 1, 1 };
		dim3 block{ this->m_block_size_update, 1, 1 };

		update_time_step_data << <grid, block >> > (
			scheduler.get_device_communication_data_ptr(),
			scheduler.get_device_time_step_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			queue_length
			);

		this->execute_initial_stage(scheduler, queue_length, run_indices_offset);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::RecursiveStage);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::RecursiveStage);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[2], scheduler.get_cuda_stream());
#endif

		this->execute_rec_stage(scheduler, queue_length, run_indices_offset);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::FinishingProcedure);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::FinishingProcedure);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[4], scheduler.get_cuda_stream());
#endif

		this->execute_fp_stage(scheduler, queue_length, run_indices_offset);

		dim3 grid{ queue_length, 1, 1 };
		dim3 block{m_block_size_local_error, 1, 1 };
		local_error_est_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_time_step_data_ptr(),
			scheduler.get_device_local_error_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_sys_size,
			queue_length // number of simulations
			);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::SpectralRadiusEstimation);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::SpectralRadiusEstimation);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[5], scheduler.get_cuda_stream());
#endif

		this->execute_spectral_radius_stage(scheduler, queue_length, run_indices_offset);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::RhsNormEstimation);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::RhsNormEstimation);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[6], scheduler.get_cuda_stream());
#endif

		dim3 grid{ queue_length, 1, 1 };
		dim3 block{ m_block_size_rhs_norm, 1, 1 };

		rhs_l2_norm_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_rhs_state_ptr(),
			scheduler.get_device_rhs_norm_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_sys_size,
			queue_length // number of simulations
			);

		CUDA_LOG(cudaGetLastError());
	}

	CUDA_LOG(cudaStreamSynchronize(scheduler.get_cuda_stream()));

	this->m_cuda_status = cudaGetLastError();
}

