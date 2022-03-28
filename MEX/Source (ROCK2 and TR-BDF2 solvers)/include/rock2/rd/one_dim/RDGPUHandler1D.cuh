#pragma once
// This class contains GPU implementations for all ROCK2 operations in the case of one-dimensional two-component reaction-diffusion systems.

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>

#include "utils/CUDALogger.cuh"

#include "rock2/R2GPUScheduler.cuh"
#include "rock2/R2GPUHandler.cuh"

#include "R2CUDAKernels.cuh"

#define LOG_TIME

template <typename value_type, typename index_type>
class RDGPUHandler1D : public R2GPUHandler<value_type, index_type>
{
private:
	index_type m_grid_resolution;

public:
	RDGPUHandler1D(
		index_type t_sim_number,
		index_type t_sys_size,
		index_type t_grid_resolution,
		value_type t_diff_coeff_1,
		value_type t_diff_coeff_2,
		const std::vector<value_type>& t_model_parameters,
		index_type t_device_number,
		value_type t_atol = 1e-3,
		value_type t_rtol = 1e-3
	);

	cudaError_t handle_computations(R2GPUScheduler<value_type, index_type>& scheduler) override;

//#ifdef LOG_TIME
	std::vector<long long> m_times;
//#endif
	
};

template<typename value_type, typename index_type>
RDGPUHandler1D<value_type, index_type>::RDGPUHandler1D(
	index_type t_sim_number,
	index_type t_sys_size,
	index_type t_grid_resolution,
	value_type t_diff_coeff_1, 
	value_type t_diff_coeff_2, 
	const std::vector<value_type>& t_model_parameters,
	index_type t_device_number,
	value_type t_atol,
	value_type t_rtol
)
	: R2GPUHandler<value_type, index_type>(t_sim_number, t_sys_size, t_model_parameters, t_device_number, t_atol, t_rtol),
	m_grid_resolution(t_grid_resolution)
{
	CUDA_LOG(cudaSetDevice(this->m_device_number));

	value_type hinv = (t_grid_resolution - 1);
	value_type nu1_hinv2 = t_diff_coeff_1 * hinv * hinv;
	value_type nu2_hinv2 = t_diff_coeff_2 * hinv * hinv;

	CUDA_LOG(cudaMemcpyToSymbol(dc_nu1_hinv2, &nu1_hinv2, sizeof(constant_type)));
	CUDA_LOG(cudaMemcpyToSymbol(dc_nu2_hinv2, &nu2_hinv2, sizeof(constant_type)));

//#ifdef LOG_TIME
	this->m_times.resize(7);
//#endif
}

template <typename index_type>
index_type find_block_number(index_type t_total_size, index_type t_block_size)
{
	return t_total_size / t_block_size
		+ (t_total_size % t_block_size != 0);
}

template<typename value_type, typename index_type>
cudaError_t RDGPUHandler1D<value_type, index_type>::handle_computations(R2GPUScheduler<value_type, index_type>& scheduler)
{
	unsigned int block_size_overwrite = 1024;
	unsigned int block_size_update = 1024;
	unsigned int block_size_initial = 64;
	unsigned int block_size_rec = 16;
	unsigned int block_size_finish = 16;
	unsigned int block_size_local_error = 64;
	unsigned int block_size_spectral_radius = 64;
	unsigned int block_size_rhs_norm = 512;

	index_type queue_length = scheduler.get_queue_length(R2OperationType::PreviousStepOverwrite);
	size_t run_indices_offset = scheduler.get_queue_start(R2OperationType::PreviousStepOverwrite);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[0], scheduler.get_cuda_stream());
#endif

		unsigned int block_number = find_block_number(static_cast<unsigned int>(this->m_sys_size * queue_length), block_size_overwrite);

		dim3 grid{ block_number, 1, 1 };
		dim3 block{ block_size_overwrite, 1, 1 };
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
		{
			unsigned int block_number = find_block_number(static_cast<unsigned int>(queue_length), block_size_update);
			dim3 grid{ block_number,1,1 };
			dim3 block{ block_size_update ,1,1};

			update_time_step_data <<<grid,block>>>(
				scheduler.get_device_communication_data_ptr(),
				scheduler.get_device_time_step_data_ptr(),
				scheduler.get_device_run_indices_ptr() + run_indices_offset,
				queue_length
			);
		}

		{
			unsigned int block_number = find_block_number(static_cast<unsigned int>(this->m_grid_resolution * queue_length), block_size_initial);
			dim3 grid{ block_number, 1, 1 };
			dim3 block{ block_size_initial, 1, 1 };

			r2_initial_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
				scheduler.get_device_sys_state_ptr(),
				scheduler.get_device_time_step_data_ptr(),
				scheduler.get_device_run_indices_ptr() + run_indices_offset,
				this->m_grid_resolution, // grid resolution
				queue_length // number of simulations
				);
		}

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::RecursiveStage);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::RecursiveStage);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[2], scheduler.get_cuda_stream());
#endif

		unsigned int block_number = find_block_number(static_cast<unsigned int>(this->m_grid_resolution * queue_length), block_size_rec);

		dim3 grid{ this->m_grid_resolution / block_size_rec, 1, queue_length };
		dim3 block{ block_size_rec, 1, 1 };

		r2_rec_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_time_step_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_grid_resolution, // grid resolution
			queue_length // number of simulations
			);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::FinishingProcedure);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::FinishingProcedure);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[4], scheduler.get_cuda_stream());
#endif

		unsigned int block_number = find_block_number(static_cast<unsigned int>(this->m_grid_resolution * queue_length), block_size_finish);

		dim3 grid{ this->m_grid_resolution / block_size_finish, 1, queue_length };
		dim3 block{ block_size_finish, 1, 1 };

		r2_fp1_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_time_step_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_grid_resolution, // grid resolution
			queue_length // number of simulations
			);
		r2_fp2_stage << <grid, block, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_time_step_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_grid_resolution, // grid resolution
			queue_length // number of simulations
			);

		dim3 grid2{ queue_length, 1, 1 };
		dim3 block2{ block_size_local_error, 1, 1 };
		local_error_est_stage << <grid2, block2, 0, scheduler.get_cuda_stream() >> > (
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

		dim3 grid2{ queue_length, 1, 1 };
		dim3 block2{ block_size_spectral_radius, 1, 1 };
		spectral_radius_est_stage << <grid2, block2, 0, scheduler.get_cuda_stream() >> > (
			scheduler.get_device_sys_state_ptr(),
			scheduler.get_device_spectral_radius_data_ptr(),
			scheduler.get_device_run_indices_ptr() + run_indices_offset,
			this->m_grid_resolution, // grid resolution
			queue_length // number of simulations
			);

		CUDA_LOG(cudaGetLastError());
	}

	queue_length = scheduler.get_queue_length(R2OperationType::RhsNormEstimation);
	run_indices_offset = scheduler.get_queue_start(R2OperationType::RhsNormEstimation);
	if (queue_length > 0)
	{
#ifdef LOG_TIME
		CUDALogger<std::chrono::nanoseconds> logger(&this->m_times[6], scheduler.get_cuda_stream());
#endif

		CUDA_LOG(cudaGetLastError());
	}

	CUDA_LOG(cudaStreamSynchronize(scheduler.get_cuda_stream()));

	auto cuda_result = cudaGetLastError();

	return cuda_result;
}
