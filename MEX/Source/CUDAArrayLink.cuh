#pragma once
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template<typename T> class HostArrayLink
{
public:
	HostArrayLink();
	HostArrayLink(T* HostData, std::size_t Size);
	~HostArrayLink();

	cudaError_t CudaLastError = cudaSuccess;
	
	std::size_t GetSize() const;
	T* GetHostDataPointer();
	T* GetElementAddress(std::size_t Position);
	cudaError_t CUDAHostRegister(unsigned int flags);
	cudaError_t CUDAHostUnregister();

private:
	std::size_t Size = 0;
	T* HostData = nullptr;
};

template<typename T> class GPUArrayLink
{
public:
	GPUArrayLink();
	GPUArrayLink(std::size_t Size, unsigned int DeviceNumber);
	~GPUArrayLink();

	cudaError_t CudaLastError = cudaSuccess;

	std::size_t GetSize() const;
	unsigned int GetDeviceNumber();
	T* GetGPUDataPointer();
	T* GetElementAddress(std::size_t Position);
	cudaError_t AllocateGPUMemory();
	cudaError_t FreeGPUMemory();
	cudaError_t AsyncCopyFromHost(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, cudaStream_t cudaStream);
	cudaError_t AsyncCopyFromHost(HostArrayLink<T> &HostLink, cudaStream_t cudaStream);
	
	cudaError_t AsyncCopyToHost(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, cudaStream_t cudaStream);
	//cudaError_t AsyncCopyToHost(HostArrayLink<T> &HostLink, cudaStream_t cudaStream);

private:
	std::size_t Size = 0;
	unsigned int DeviceNumber = 0;
	T* GPUData = nullptr;
	cudaError_t AsyncCopy(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, unsigned int DeviceNumber, cudaStream_t cudaStream, cudaMemcpyKind kind);
};

/*  Host Array Link Implementation  */

template<typename T>
HostArrayLink<T>::HostArrayLink()
{
	this->Size = 0;
}

template<typename T>
HostArrayLink<T>::HostArrayLink(T* HostData, std::size_t Size)
{
	this->HostData = HostData;
	this->Size = Size;
}

template<typename T>
HostArrayLink<T>::~HostArrayLink()
{
}


template<typename T>
cudaError_t HostArrayLink<T>::CUDAHostRegister(unsigned int flags)
{
	this->CudaLastError = cudaHostRegister(this->HostData, this->Size * sizeof(T), flags);
	return this->CudaLastError;
}

template<typename T>
cudaError_t HostArrayLink<T>::CUDAHostUnregister()
{
	this->CudaLastError = cudaHostUnregister(this->HostData);
	return this->CudaLastError;
}

template<typename T>
T* HostArrayLink<T>::GetElementAddress(std::size_t Position)
{
	if (Position >= this->Size)
		throw std::range_error("The address requested is out of range");
	return this->HostData + Position;
}

template<typename T>
std::size_t HostArrayLink<T>::GetSize() const
{
	return this->Size;
}

template<typename T>
T * HostArrayLink<T>::GetHostDataPointer()
{
	return this->HostData;
}

/*  GPU Array Link Implementation  */

template<typename T>
GPUArrayLink<T>::GPUArrayLink()
{
	this->Size = 0;
}

template<typename T>
GPUArrayLink<T>::GPUArrayLink(std::size_t Size, unsigned int DeviceNumber)
{
	this->Size = Size;
	this->DeviceNumber = DeviceNumber;
}


template<typename T>
GPUArrayLink<T>::~GPUArrayLink()
{
	//if (this->GPUData != nullptr)
	//	throw std::exception("The GPU memory was not released properly!");
}

template<typename T>
std::size_t GPUArrayLink<T>::GetSize() const
{
	return this->Size;
}

template<typename T>
unsigned int GPUArrayLink<T>::GetDeviceNumber()
{
	return this->DeviceNumber;
}

template<typename T>
T* GPUArrayLink<T>::GetGPUDataPointer()
{
	return this->GPUData;
}

template<typename T>
T * GPUArrayLink<T>::GetElementAddress(std::size_t Position)
{
	if (Position >= this->Size)
		throw std::range_error("The address requested was out of range");
	return this->GPUData + Position;
}


template<typename T>
cudaError_t GPUArrayLink<T>::AllocateGPUMemory()
{
	if (this->GPUData != nullptr)
		throw std::runtime_error("The memory has been already allocated!");

	if (this->Size == 0)
		throw std::runtime_error("Cannot allocate zero bytes of memory!");

	this->CudaLastError = cudaSetDevice(this->DeviceNumber);
	if (this->CudaLastError != cudaSuccess)
		return this->CudaLastError;

	this->CudaLastError = cudaMalloc(&this->GPUData, this->Size * sizeof(T));
	return this->CudaLastError;
}

template<typename T>
cudaError_t GPUArrayLink<T>::FreeGPUMemory()
{
	this->CudaLastError = cudaSetDevice(this->DeviceNumber);
	if (this->CudaLastError != cudaSuccess)
		return this->CudaLastError;

	this->CudaLastError = cudaFree(this->GPUData);
	if (this->CudaLastError == cudaSuccess)
		this->GPUData = nullptr;
	return this->CudaLastError;
}

template<typename T>
cudaError_t GPUArrayLink<T>::AsyncCopyFromHost(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, cudaStream_t cudaStream)
{
	this->CudaLastError = this->AsyncCopy(HostLink, GPUStartIndex, HostStartIndex, Size, this->DeviceNumber, cudaStream, cudaMemcpyHostToDevice);
	return this->CudaLastError;
}

template<typename T>
cudaError_t GPUArrayLink<T>::AsyncCopyFromHost(HostArrayLink<T> &HostLink, cudaStream_t cudaStream)
{
	this->CudaLastError = this->AsyncCopyFromHost(HostLink, 0, 0, HostLink->Size, this->DeviceNumber, cudaStream);
	return this->CudaLastError;
}

template<typename T>
cudaError_t GPUArrayLink<T>::AsyncCopyToHost(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, cudaStream_t cudaStream)
{
	this->CudaLastError = this->AsyncCopy(HostLink, GPUStartIndex, HostStartIndex, Size, this->DeviceNumber, cudaStream, cudaMemcpyDeviceToHost);
	return this->CudaLastError;
}

/*
template<typename T>
cudaError_t GPUArrayLink<T>::AsyncCopyToHost(HostArrayLink<T> &HostLink, cudaStream_t cudaStream)
{
	this->CudaLastError = this->AsyncCopy(HostLink, GPUStartIndex, HostStartIndex, Size, this->DeviceNumber, cudaStream, cudaMemcpyDeviceToHost);
	return this->CudaLastError;
}
*/

template<typename T>
cudaError_t GPUArrayLink<T>::AsyncCopy(HostArrayLink<T> &HostLink, std::size_t GPUStartIndex, std::size_t HostStartIndex, std::size_t Size, unsigned int DeviceNumber, cudaStream_t cudaStream, cudaMemcpyKind kind)
{
	auto trace_error = [this, HostLink, GPUStartIndex, HostStartIndex, Size, DeviceNumber](std::string message) -> std::string
	{
		std::ostringstream os;
		os << "GPUArray::AsyncCopy error: " << message << " "
			<< "Device array size: " << this->Size << " elements. "
			<< "Requested size: " << Size << " elements. "
			<< "Requested device start index: " << GPUStartIndex << ". "
			<< "Requested host start index: " << HostStartIndex << ". "
			<< "Host size: " << HostLink.GetSize() << " elements.";
		return os.str();
	};

	if (GPUStartIndex + Size > this->Size)
	{	
		auto error_message = trace_error("Device start index is not correct!");
		throw std::runtime_error(&error_message[0]);
	}

	if (HostStartIndex + Size > HostLink.GetSize())
	{
		auto error_message = trace_error("Host array address is not correct!");
		throw std::runtime_error(&error_message[0]);
	}

	if (this->GPUData == nullptr)
	{
		auto error_message = trace_error("GPU pointer cannot be null!");
		throw std::runtime_error(&error_message[0]);
	}

	this->CudaLastError = cudaSetDevice(DeviceNumber);
	if (this->CudaLastError != cudaSuccess)
		return this->CudaLastError;

	if (kind == cudaMemcpyDeviceToHost)
		this->CudaLastError = cudaMemcpyAsync(HostLink.GetElementAddress(HostStartIndex), this->GetElementAddress(GPUStartIndex),
			Size * sizeof(T), kind, cudaStream);
	else if (kind == cudaMemcpyHostToDevice)
		this->CudaLastError = cudaMemcpyAsync(this->GetElementAddress(GPUStartIndex), HostLink.GetElementAddress(HostStartIndex),
			Size * sizeof(T), kind, cudaStream);
	else throw std::runtime_error("GPUArray::AsyncCopy error: Operation not supported!");

	return this->CudaLastError;
}


