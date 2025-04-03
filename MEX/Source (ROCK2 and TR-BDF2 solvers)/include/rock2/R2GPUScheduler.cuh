// This class implements the connection interface for GPU computations.
#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>

#include "rock2/R2Scheduler.h"
#include "utils/MemoryLink.h"
#include "utils/CUDAArrayLinks_v4.cuh"

#include <string>
#include "utils/TestUtils.h"

#define CUDA_LOG(op){auto res = op;  if (res != cudaSuccess){std::cerr << "FILE " << __FILE__ << " LINE " << __LINE__ << " CUDA ERROR: " << res << std::endl;}}



template <typename value_type, typename index_type>
class R2GPUScheduler : public R2Scheduler<value_type, index_type>
{
private:
	cudaStream_t m_cuda_stream;
	index_type m_device_number;
	HostArrayLink<value_type> m_host_input_data;
	HostArrayLink<value_type> m_host_output_data;
	HostArrayLink<index_type> m_host_run_indices;
	HostArrayLink<value_type> m_host_communication_data; // used for transfering time steps ONLY!

	HostArrayLink<value_type> m_host_scalar_data;

	DeviceArrayLink<value_type> m_device_sys_state_data;
	DeviceArrayLink<index_type> m_device_run_indices;
	DeviceArrayLink<value_type> m_device_communication_data; // used for transfering time steps ONLY!
	DeviceArrayLink<value_type> m_device_rhs_state_data; // used to store the r.h.s., evaluated during the initial stage procedure

	DeviceArrayLink<value_type> m_device_scalar_data;

	DeviceArrayLink<value_type> m_device_time_step_data;

public:
	R2GPUScheduler(
		index_type t_sim_number,
		MemoryLink<value_type>& t_input_data,
		MemoryLink<value_type>& t_output_data,
		const index_type t_device_number);
	~R2GPUScheduler();

	value_type get_local_error(index_type t_queue_number) override;
	value_type get_spectral_radius(index_type t_queue_number) override;
	value_type get_rhs_norm(index_type t_queue_number) override;

	cudaStream_t get_cuda_stream();
	value_type* get_device_communication_data_ptr();
	value_type* get_device_time_step_data_ptr();
	value_type* get_device_local_error_data_ptr();
	value_type* get_device_spectral_radius_data_ptr();
	value_type* get_device_rhs_norm_data_ptr();
	index_type* get_device_run_indices_ptr();
	value_type* get_device_sys_state_ptr();
	value_type* get_device_rhs_state_ptr();
	size_t get_queue_start(R2OperationType t_operation);
	size_t get_queue_length(R2OperationType t_operation);
	
	void send_data_to_device();
	void receive_data_from_device();

	void finalize();

	void log_sys_state(std::string t_file_name) override;

};

 template<typename value_type, typename index_type>
 void R2GPUScheduler<value_type, index_type>::log_sys_state(std::string t_file_name)
{
	std::vector<value_type> temp_data(m_device_sys_state_data.size());
	m_device_sys_state_data.copy_to_host(temp_data);
	write_vector(t_file_name, temp_data);
}

template<typename value_type, typename index_type>
R2GPUScheduler<value_type, index_type>::R2GPUScheduler(
	index_type t_sim_number,
	MemoryLink<value_type>& t_input_data, 
	MemoryLink<value_type>& t_output_data, 
	const index_type t_device_number
) : R2Scheduler<value_type, index_type>(t_sim_number),
m_device_number(t_device_number),
m_host_input_data(t_input_data),
m_host_output_data(t_output_data),
m_host_run_indices({ this->m_operation_queue.data(), this->m_operation_queue.size() }),
m_host_communication_data(this->m_time_step_queue),
m_host_scalar_data(3 * t_sim_number),
m_device_sys_state_data(4 * t_input_data.size(), t_device_number), // y, yprev, yjm1, yjm2
m_device_run_indices(this->m_host_run_indices.size(), t_device_number),
m_device_communication_data(this->m_host_communication_data.size(), t_device_number),
m_device_rhs_state_data(t_input_data.size(), t_device_number),
m_device_scalar_data(this->m_host_scalar_data.size(), t_device_number),
m_device_time_step_data(t_sim_number, t_device_number)
{
	CUDA_LOG(cudaSetDevice(this->m_device_number));
	CUDA_LOG(cudaStreamCreate(&this->m_cuda_stream));

	// host array link registration
	CUDA_LOG(this->m_host_input_data.cuda_host_register(cudaHostRegisterPortable));
	if (this->m_host_output_data.data() != this->m_host_input_data.data())
		CUDA_LOG(this->m_host_output_data.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(this->m_host_run_indices.cuda_host_register(cudaHostRegisterPortable));
	CUDA_LOG(this->m_host_communication_data.cuda_host_register(cudaHostRegisterPortable));

	CUDA_LOG(this->m_host_scalar_data.cuda_host_register(cudaHostRegisterPortable));


	// copying initial data to device
	auto sys_size = t_input_data.size() / t_sim_number;
	HostArrayLink<value_type> temp_host_data(4 * sys_size * t_sim_number);
	temp_host_data.cuda_host_register(cudaHostRegisterPortable);
	for (auto k = 0u; k < t_sim_number; ++k)
	{
		/*
		cudaMemcpy(
			temp_host_data.data() + 4 * k * sys_size, 
			t_input_data.data() + k * sys_size, sys_size * sizeof(value_type), 
			cudaMemcpyHostToHost
		);
		*/
		std::copy(
			t_input_data.begin() + k * sys_size,
			t_input_data.begin() + (k + 1) * sys_size,
			temp_host_data.begin() + 4 * k * sys_size
		);
	}

	CUDA_LOG(this->m_device_sys_state_data.copy_from_host_async(temp_host_data, this->m_cuda_stream));
	CUDA_LOG(cudaStreamSynchronize(this->m_cuda_stream));

	temp_host_data.cuda_host_unregister();
}

template<typename value_type, typename index_type>
R2GPUScheduler<value_type, index_type>::~R2GPUScheduler()
{
	// host array link cleanup
	CUDA_LOG(this->m_host_input_data.cuda_host_unregister());
	if (this->m_host_output_data.data() != this->m_host_input_data.data())
		this->m_host_output_data.cuda_host_unregister();
	CUDA_LOG(this->m_host_run_indices.cuda_host_unregister());
	CUDA_LOG(this->m_host_communication_data.cuda_host_unregister());
	CUDA_LOG(this->m_host_scalar_data.cuda_host_unregister());

	CUDA_LOG(cudaStreamDestroy(this->m_cuda_stream));
}

template<typename value_type, typename index_type>
value_type R2GPUScheduler<value_type, index_type>::get_local_error(index_type t_queue_number)
{
	return value_type(*this->m_host_scalar_data[t_queue_number]);
}

template<typename value_type, typename index_type>
value_type R2GPUScheduler<value_type, index_type>::get_spectral_radius(index_type t_queue_number)
{
	return value_type(*this->m_host_scalar_data[this->m_sim_number + t_queue_number]);
}

template<typename value_type, typename index_type>
value_type R2GPUScheduler<value_type, index_type>::get_rhs_norm(index_type t_queue_number)
{
	return value_type(*this->m_host_scalar_data[2 * this->m_sim_number + t_queue_number]);
}

template<typename value_type, typename index_type>
cudaStream_t R2GPUScheduler<value_type, index_type>::get_cuda_stream()
{
	return this->m_cuda_stream;
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_communication_data_ptr()
{
	return this->m_device_communication_data.data();
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_time_step_data_ptr()
{
	return this->m_device_time_step_data.data();
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_local_error_data_ptr()
{
	return this->m_device_scalar_data.data();
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_spectral_radius_data_ptr()
{
	return this->m_device_scalar_data.data() + this->m_sim_number;
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_rhs_norm_data_ptr()
{
	return this->m_device_scalar_data.data() + 2 * this->m_sim_number;
}

template<typename value_type, typename index_type>
index_type* R2GPUScheduler<value_type, index_type>::get_device_run_indices_ptr()
{
	return this->m_device_run_indices.data();
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_sys_state_ptr()
{
	return this->m_device_sys_state_data.data();
}

template<typename value_type, typename index_type>
value_type* R2GPUScheduler<value_type, index_type>::get_device_rhs_state_ptr()
{
	return this->m_device_rhs_state_data.data();
}

template<typename value_type, typename index_type>
size_t R2GPUScheduler<value_type, index_type>::get_queue_start(R2OperationType t_operation)
{
	return this->m_operation_queue.get_operation_queue_start(t_operation);
}

template<typename value_type, typename index_type>
size_t R2GPUScheduler<value_type, index_type>::get_queue_length(R2OperationType t_operation)
{
	return this->m_operation_queue.get_operation_queue_length(t_operation);
}

template<typename value_type, typename index_type>
void R2GPUScheduler<value_type, index_type>::send_data_to_device()
{
	CUDA_LOG(this->m_device_run_indices.copy_from_host_async(this->m_host_run_indices, this->m_cuda_stream));

	if (this->get_queue_length(R2OperationType::InitialStage) > 0) // schedule initial stage
	{
		CUDA_LOG(this->m_device_communication_data.copy_from_host_async(
			this->m_host_communication_data,
			0, 
			0, 
			this->get_queue_length(R2OperationType::InitialStage), 
			this->m_cuda_stream
		));
	}

	CUDA_LOG(cudaStreamSynchronize(this->m_cuda_stream));
}

template<typename value_type, typename index_type>
void R2GPUScheduler<value_type, index_type>::receive_data_from_device()
{
	if ((this->get_queue_length(R2OperationType::FinishingProcedure) > 0) 
		|| (this->get_queue_length(R2OperationType::SpectralRadiusEstimation) > 0) 
		|| (this->get_queue_length(R2OperationType::RhsNormEstimation) > 0))
	{
		CUDA_LOG(this->m_device_scalar_data.copy_to_host_async(this->m_host_scalar_data, this->m_cuda_stream));
		CUDA_LOG(cudaStreamSynchronize(this->m_cuda_stream));
	}
}

template<typename value_type, typename index_type>
void R2GPUScheduler<value_type, index_type>::finalize()
{
	auto sys_size = this->m_host_output_data.size() / this->m_sim_number;
	HostArrayLink<value_type> temp_host_data(4 * sys_size * this->m_sim_number);
	temp_host_data.cuda_host_register(cudaHostRegisterPortable);

	CUDA_LOG(this->m_device_sys_state_data.copy_to_host_async(temp_host_data, 0, 0, temp_host_data.size(), this->m_cuda_stream));

	CUDA_LOG(cudaStreamSynchronize(this->m_cuda_stream));

	for (auto k = 0u; k < this->m_sim_number; ++k)
	{
		/*
		cudaMemcpy(
			m_host_output_data.data() + k * sys_size, 
			temp_host_data.data() + 4 * k * sys_size, 
			sys_size * sizeof(value_type), 
			cudaMemcpyHostToHost
		);
		*/
		std::copy(
			temp_host_data.begin() + 4 * k * sys_size + sys_size,
			temp_host_data.begin() + 4 * k * sys_size + 2 * sys_size, 
			this->m_host_output_data.data() + k * sys_size);
	}

	temp_host_data.cuda_host_unregister();

	CUDA_LOG(cudaStreamSynchronize(this->m_cuda_stream));
}
