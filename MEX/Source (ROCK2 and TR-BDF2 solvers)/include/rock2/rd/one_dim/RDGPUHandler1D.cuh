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

protected:

	unsigned int m_block_size_initial = 64;
	unsigned int m_block_size_rec = 64;
	unsigned int m_block_size_finish = 64;
	unsigned int m_block_size_spectral_radius = 64;

	void execute_initial_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) override;
	void execute_rec_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) override;
	void execute_fp_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) override;
	void execute_spectral_radius_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset) override;

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
}

template<typename value_type, typename index_type>
void RDGPUHandler1D<value_type, index_type>::execute_initial_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset)
{
	unsigned int block_number = this->find_block_number(
		static_cast<unsigned int>(this->m_grid_resolution * queue_length),
		this->m_block_size_initial
	);

	dim3 grid{ block_number, 1, 1 };
	dim3 block{ m_block_size_initial, 1, 1 };

	r2_initial_stage<value_type, index_type> << <grid, block, 0, scheduler.get_cuda_stream() >> > (
		scheduler.get_device_sys_state_ptr(),
		scheduler.get_device_time_step_data_ptr(),
		scheduler.get_device_run_indices_ptr() + run_indices_offset,
		this->m_grid_resolution, // grid resolution
		queue_length // number of simulations
		);
}

template<typename value_type, typename index_type>
void RDGPUHandler1D<value_type, index_type>::execute_rec_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset)
{
	unsigned int block_number = this->find_block_number(
		static_cast<unsigned int>(this->m_grid_resolution),
		this->m_block_size_rec
	);

	dim3 grid{ block_number, 1, queue_length };
	dim3 block{ this->m_block_size_rec, 1, 1 };

	r2_rec_stage<value_type, index_type> << <grid, block, 0, scheduler.get_cuda_stream() >> > (
		scheduler.get_device_sys_state_ptr(),
		scheduler.get_device_time_step_data_ptr(),
		scheduler.get_device_run_indices_ptr() + run_indices_offset,
		this->m_grid_resolution, // grid resolution
		queue_length // number of simulations
		);
}

template<typename value_type, typename index_type>
void RDGPUHandler1D<value_type, index_type>::execute_fp_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset)
{
	unsigned int block_number = this->find_block_number(
		static_cast<unsigned int>(this->m_grid_resolution),
		this->m_block_size_finish);

	dim3 grid{ block_number, 1, queue_length };
	dim3 block{ this->m_block_size_finish, 1, 1 };

	r2_fp1_stage<value_type, index_type> << <grid, block, 0, scheduler.get_cuda_stream() >> > (
		scheduler.get_device_sys_state_ptr(),
		scheduler.get_device_time_step_data_ptr(),
		scheduler.get_device_run_indices_ptr() + run_indices_offset,
		this->m_grid_resolution, // grid resolution
		queue_length // number of simulations
		);

	r2_fp2_stage<value_type, index_type> << <grid, block, 0, scheduler.get_cuda_stream() >> > (
		scheduler.get_device_sys_state_ptr(),
		scheduler.get_device_time_step_data_ptr(),
		scheduler.get_device_run_indices_ptr() + run_indices_offset,
		this->m_grid_resolution, // grid resolution
		queue_length // number of simulations
		);
}

template<typename value_type, typename index_type>
void RDGPUHandler1D<value_type, index_type>::execute_spectral_radius_stage(R2GPUScheduler<value_type, index_type>& scheduler, index_type queue_length, size_t run_indices_offset)
{
	dim3 grid{ queue_length, 1, 1 };
	dim3 block{ this->m_block_size_spectral_radius, 1, 1 };
	spectral_radius_est_stage<value_type, index_type> << <grid, block, 0, scheduler.get_cuda_stream() >> > (
		scheduler.get_device_sys_state_ptr(),
		scheduler.get_device_spectral_radius_data_ptr(),
		scheduler.get_device_run_indices_ptr() + run_indices_offset,
		this->m_grid_resolution, // grid resolution
		queue_length // number of simulations
		);
}

