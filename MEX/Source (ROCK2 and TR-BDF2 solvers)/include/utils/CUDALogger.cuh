#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>

#include "utils/RunResult.cuh"

template <typename time_type>
class CUDALogger
{
public:
	CUDALogger(long long* t_time, cudaStream_t t_cuda_stream = 0)
		:m_time(t_time), m_cuda_stream(t_cuda_stream)
	{
		m_start = std::chrono::high_resolution_clock::now();
	}
	~CUDALogger()
	{
		if (m_cuda_stream == 0)
		{
			CUDA_LOG(cudaDeviceSynchronize());
		}
		else
		{
			CUDA_LOG(cudaStreamSynchronize(m_cuda_stream));
		}

		m_end = std::chrono::high_resolution_clock::now();
		long long duration = std::chrono::duration_cast<time_type>(m_end - m_start).count();
		*m_time += duration;
	}
private:
	long long* m_time;
	cudaStream_t m_cuda_stream;
	std::chrono::high_resolution_clock::time_point m_start;
	std::chrono::high_resolution_clock::time_point m_end;
};
