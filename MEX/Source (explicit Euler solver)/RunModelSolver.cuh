#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "CUDAArrayLink.cuh"

struct RunResult
{
	std::string message;
	cudaError_t code;

	RunResult(cudaError_t _code, std::string _message)
		: code(_code), message(_message) {};
};

std::ostream& operator << (std::ostream& os, const RunResult& res)
{
	os << "code: " << res.code << ", message: " << res.message;
	return os;
}

struct LaunchDataContainer
{
	size_t ThreadsPerBlock;
	size_t BlockNumber;
	size_t SharedMemoryPerBlock;
};

template<typename value_type> struct SolverParameters
{
	value_type T0;
	value_type T1;
	value_type dt;
};

// declarations

template<typename value_type> struct DeviceModelParameters;

template <typename value_type> __device__ inline void ModelSolver(
	value_type* data,
	value_type* sm,
	const DeviceModelParameters<value_type>& par,
	value_type T0,
	value_type T1,
	value_type dT
);

// end declarations

template<typename value_type> __global__ void cudaKernel(
	value_type* data,
	const DeviceModelParameters<value_type> par,
	const SolverParameters<value_type> solpar)
{
	extern __shared__ value_type sm[];
	ModelSolver(data, sm, par, solpar.T0, solpar.T1, solpar.dt);
}

#define CUDA_CALL(op,message) {cudaError_t res = op;  if (res != cudaSuccess) { return RunResult(res, message);}}

void ReportDuration(std::chrono::time_point<std::chrono::high_resolution_clock> start, std::string message)
{
	auto now = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
	std::cout << message << " Elapsed time: " << duration << "ms. " << std::endl;
}


template <typename value_type> RunResult RunModelSolver(
	const LaunchDataContainer& launchdata, // data, which determines how CUDA kernel call should be configured
	value_type* initialdata, // pointer to the array, where initial data for simulations is stored
	const size_t size, // size (in elements) of the initialdata array
	value_type* output, // pointer to the array, where final data should be stored. Assumed to have the same size as initialdata
	const DeviceModelParameters<value_type>& par,
	const SolverParameters<value_type>& solpar,
	const std::vector<int>& devices,
	bool showTiming = true
)
{

	size_t ElementsPerSystem = static_cast<unsigned int>(size / launchdata.BlockNumber);
	size_t BlocksPerDevice = static_cast<unsigned int>(launchdata.BlockNumber / devices.size());
	size_t ElementsPerDevice = BlocksPerDevice * ElementsPerSystem;

	std::vector<size_t> DeviceElementNumber(devices.size(), ElementsPerDevice);
	std::vector<size_t> DeviceBlockNumber(devices.size(), BlocksPerDevice);

	DeviceBlockNumber[0] += (launchdata.BlockNumber - BlocksPerDevice * devices.size());
	DeviceElementNumber[0] += (launchdata.BlockNumber - BlocksPerDevice * devices.size()) * ElementsPerSystem;
	
	
	// CUDA streams
	std::vector<cudaStream_t> CudaStreams(devices.size());
	// host links
	HostArrayLink<value_type> HostInitialData(initialdata, size);
	HostArrayLink<value_type> HostOutputData(output, size);
	// device(s) links
	std::vector<std::shared_ptr<GPUArrayLink<value_type>>> DeviceInitialData(devices.size());

	auto start = std::chrono::high_resolution_clock::now();
	if (showTiming)
		std::cout << "Beginning timing count..." << std::endl;

	// host memory allocation
	CUDA_CALL(cudaSetDevice(devices[0]), "cudaSetDevice error!");
	CUDA_CALL(HostInitialData.CUDAHostRegister(cudaHostRegisterPortable), "Initial data host allocation failed!");
	CUDA_CALL(HostOutputData.CUDAHostRegister(cudaHostRegisterPortable), "Output data host allocation failed!");

	// synchronous part
	for (unsigned int i = 0; i < devices.size(); ++i)
	{
		// stream creation
		CUDA_CALL(cudaSetDevice(devices[i]), "cudaSetDevice error!");
		CUDA_CALL(cudaStreamCreate(&CudaStreams[i]), "cudaStreamCreate failed!");

		// memory initialization on device
		DeviceInitialData[i] = std::make_shared<GPUArrayLink<value_type>>(DeviceElementNumber[i], devices[i]);
		CUDA_CALL(DeviceInitialData[i]->AllocateGPUMemory(), "Device memory allocation failed!");
	}

	if (showTiming)
		ReportDuration(start, "Memory allocation on all devices is finished.");
		
	size_t DeviceOffset = 0;

	// asynchronous part
	for (unsigned int i = 0; i < devices.size(); ++i)
	{
		
		// copying data from host to device
		CUDA_CALL(DeviceInitialData[i]->AsyncCopyFromHost(
			HostInitialData,
			0, // device start index
			DeviceOffset, // host start index
			DeviceElementNumber[i], // number of elements
			CudaStreams[i]),
			"CUDA copy from host to device failed!");		

		if (showTiming)
			ReportDuration(start, "AsyncCopyFromHost executed.");
			

		// MAIN COMPUTATIONS GO HERE!
		dim3 grid{ static_cast<unsigned int>(DeviceBlockNumber[i]), 1, 1 };
		dim3 block{ static_cast<unsigned int>(launchdata.ThreadsPerBlock), 1, 1 };

		CUDA_CALL(cudaSetDevice(devices[i]), "cudaSetDevice error!");

		
		cudaKernel<value_type> << < grid, block, launchdata.SharedMemoryPerBlock * sizeof(value_type), CudaStreams[i] >> > (
			DeviceInitialData[i]->GetGPUDataPointer(), // initial data
			par, // model parameters,
			solpar // solver parameters
			);

		if (showTiming)
			ReportDuration(start, "cudaKernel launched.");
			

		CUDA_CALL(cudaGetLastError(), "CUDA kernel execution failed!");

		
		// copying data from device to host
		CUDA_CALL(DeviceInitialData[i]->AsyncCopyToHost(
			HostOutputData,
			0, // device start index
			DeviceOffset, // host start index
			DeviceElementNumber[i], // number of elements
			CudaStreams[i]),
			"CUDA copy from device to host failed!");

		if (showTiming)
			ReportDuration(start, "AsyncCopyToHost executed.");
			
		// only at the end of asynchronous part (after the second copy has been arranged) we increase the device offset
		DeviceOffset += DeviceElementNumber[i];
			
	}

	// Device syncronization
	for (unsigned int i = 0; i < devices.size(); ++i)
	{
		CUDA_CALL(cudaSetDevice(devices[i]), "cudaSetDevice error!");
		CUDA_CALL(cudaStreamSynchronize(CudaStreams[i]), "CUDA stream synchronization failed!");
		//CUDA_CALL(cudaDeviceSynchronize(), "CUDA device synchronization failed!");

		if (showTiming)
			ReportDuration(start, "Syncronization completed.");
	}

	//syncronous part
	// host cleanup
	CUDA_CALL(cudaSetDevice(devices[0]), "cudaSetDevice error!");
	CUDA_CALL(HostInitialData.CUDAHostUnregister(), "Host initial data memory deallocation failed!");
	CUDA_CALL(HostOutputData.CUDAHostUnregister(),"Host output data memory deallocation failed!");

	// device cleanup
	for (unsigned int i = 0; i < devices.size(); ++i)
	{
		// free device memory
		CUDA_CALL(DeviceInitialData[i]->FreeGPUMemory(), "CUDA free memory error!");

		CUDA_CALL(cudaSetDevice(devices[i]), "cudaSetDevice error!");

		// CUDA stream destroying
		CUDA_CALL(cudaStreamDestroy(CudaStreams[i]), "CUDA stream destruction error!");

		// CUDA device reset
		CUDA_CALL(cudaDeviceReset(), "CUDA device reset failed!");
	} // for

	if (showTiming)
		ReportDuration(start, "Cleanup completed.");

	return RunResult(cudaSuccess, "");
}
