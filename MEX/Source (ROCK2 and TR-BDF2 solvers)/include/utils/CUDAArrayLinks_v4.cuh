#pragma once

#include <exception>
#include <iostream>
#include <vector>
#include <sstream>
#include <stdexcept>
#include <string>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MemoryLink.h"

template<typename T> class HostArrayLink
{
public:
	HostArrayLink();
	HostArrayLink(T* t_host_data, std::size_t t_size);
	HostArrayLink(MemoryLink<T>& t_link);
	HostArrayLink(std::vector<T>& t_vector);
	HostArrayLink(std::size_t t_size);
	~HostArrayLink();

	cudaError_t cuda_last_error();
	std::size_t size() const;
	T* data();
	T* operator[](const std::size_t t_position);
	cudaError_t cuda_host_register(std::size_t t_flags);
	cudaError_t cuda_host_unregister();

	T* begin();
	T* end();
	const T* begin() const;
	const T* end() const;

private:
	std::size_t m_size = 0;
	T* m_host_data = nullptr;
	cudaError_t m_cuda_last_error = cudaSuccess;
	std::vector<T> m_own_data;
};

template<typename T> class DeviceArrayLink
{
public:
	DeviceArrayLink();
	DeviceArrayLink(std::size_t t_size, unsigned int t_device_number);
	DeviceArrayLink(std::size_t t_size);
	~DeviceArrayLink();

	std::size_t size() const;
	unsigned int device_number();
	T* data();
	T* operator[](std::size_t t_position);
	operator T*() const;

	cudaError_t allocate();
	cudaError_t free();
	cudaError_t copy_from_host_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, cudaStream_t t_cuda_stream);
	cudaError_t copy_from_host_async(HostArrayLink<T> &t_host_link, cudaStream_t t_cuda_stream);
	cudaError_t copy_to_host_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, cudaStream_t t_cuda_stream);
	cudaError_t copy_to_host_async(HostArrayLink<T> &t_host_link, cudaStream_t t_cuda_stream);

	cudaError_t copy_from_host(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size);
	cudaError_t copy_from_host(HostArrayLink<T> &t_host_link);
	cudaError_t copy_to_host(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size);

	cudaError_t copy_from_host(T* t_host_data, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size);
	cudaError_t copy_from_host(T* t_host_data, std::size_t t_size);
	cudaError_t copy_to_host(T* t_host_data, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size);
	cudaError_t copy_to_host(T* t_host_data, std::size_t t_size);

	cudaError_t copy_from_host(std::vector<T> &t_host_data);
	cudaError_t copy_to_host(std::vector<T> &t_host_data);

	cudaError_t copy_to_device(DeviceArrayLink<T>& t_device_data);
	cudaError_t copy_to_device_async(DeviceArrayLink<T>& t_device_data, cudaStream_t t_cuda_stream);
	cudaError_t copy_to_device_async(DeviceArrayLink<T>& t_device_data, std::size_t t_src_start_index, std::size_t t_dst_start_index, std::size_t t_size, cudaStream_t cuda_stream);

	cudaError_t cuda_last_error();
private:
	cudaError_t m_cuda_last_error = cudaSuccess;
	std::size_t m_size = 0;
	unsigned int m_device_number = 0;
	T* m_device_data = nullptr;
	cudaError_t copy_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, unsigned int t_device_number, cudaStream_t t_cuda_stream, cudaMemcpyKind t_kind);
	cudaError_t copy_async(DeviceArrayLink<T> &t_device_link, std::size_t t_src_start_index, std::size_t t_dst_start_index, std::size_t t_size, cudaStream_t t_cuda_stream);
	cudaError_t copy(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, unsigned int t_device_number, cudaMemcpyKind t_kind);
};

//  Host Array Link Implementation 

template<typename T> inline T* HostArrayLink<T>::begin()
{
	return this->m_host_data;
}

template<typename T> inline T* HostArrayLink<T>::end()
{
	return this->m_host_data + m_size;
}

template<typename T> inline const T* HostArrayLink<T>::begin() const
{
	return this->m_host_data;
}

template<typename T> inline const T* HostArrayLink<T>::end() const
{
	return this->m_host_data + m_size;
}

template<typename T>
HostArrayLink<T>::HostArrayLink()
{
	this->m_size = 0;
}

template<typename T>
HostArrayLink<T>::HostArrayLink(T* t_host_data, std::size_t t_size)
{
	this->m_host_data = t_host_data;
	this->m_size = t_size;
}

template<typename T>
HostArrayLink<T>::HostArrayLink(MemoryLink<T>& t_link)
{
	this->m_host_data = t_link.data();
	this->m_size = t_link.size();
}

template<typename T>
HostArrayLink<T>::HostArrayLink(std::vector<T>& t_vector)
	:HostArrayLink<T>(t_vector.data(), t_vector.size())
{
}

template<typename T>
HostArrayLink<T>::HostArrayLink(std::size_t t_size)
	: m_own_data(t_size)
{
	this->m_host_data = m_own_data.data();
	this->m_size = t_size;
}

template<typename T>
HostArrayLink<T>::~HostArrayLink()
{
}

template<typename T>
cudaError_t HostArrayLink<T>::cuda_host_register(std::size_t t_flags)
{
	this->m_cuda_last_error = cudaHostRegister(this->m_host_data, this->m_size * sizeof(T), static_cast<unsigned int>(t_flags));
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t HostArrayLink<T>::cuda_host_unregister()
{
	this->m_cuda_last_error = cudaHostUnregister(this->m_host_data);
	return this->m_cuda_last_error;
}

template<typename T>
T* HostArrayLink<T>::operator[](std::size_t t_position)
{
	if (t_position >= this->m_size)
		throw std::range_error("HostArrayLink<T>::operator[] error: The address requested is out of range");
	return this->m_host_data + t_position;
}

template<typename T>
cudaError_t HostArrayLink<T>::cuda_last_error()
{
	return cudaError_t(this->m_cuda_last_error);
}

template<typename T>
std::size_t HostArrayLink<T>::size() const
{
	return this->m_size;
}

template<typename T>
T * HostArrayLink<T>::data()
{
	return this->m_host_data;
}



//  GPU Array Link Implementation 

template<typename T>
DeviceArrayLink<T>::DeviceArrayLink()
{
	this->m_device_data = nullptr;
	this->m_size = 0;
}

template<typename T>
DeviceArrayLink<T>::DeviceArrayLink(std::size_t t_size, unsigned int t_device_number)
{
	this->m_size = t_size;
	this->m_device_number = t_device_number;
	this->allocate();
}

template<typename T> 
DeviceArrayLink<T>::DeviceArrayLink(std::size_t t_size)
	:DeviceArrayLink<T>(t_size, 0){}


template<typename T>
DeviceArrayLink<T>::~DeviceArrayLink()
{
	if (this->m_device_data != nullptr)
		this->free();
}

template<typename T>
std::size_t DeviceArrayLink<T>::size() const
{
	return this->m_size;
}

template<typename T>
unsigned int DeviceArrayLink<T>::device_number()
{
	return this->m_device_number;
}

template<typename T>
T* DeviceArrayLink<T>::data()
{
	return this->m_device_data;
}

template<typename T>
T* DeviceArrayLink<T>::operator[](std::size_t t_position)
{
	if (t_position >= this->m_size)
		throw std::range_error("GPUArrayLink<T>::operator[] error: The address requested was out of range");
	return this->m_device_data + t_position;
}

template<typename T>
DeviceArrayLink<T>::operator T* () const
{
	return this->m_device_data;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::allocate()
{
	if (this->m_device_data != nullptr)
		throw std::runtime_error("GPUArrayLink<T>::allocate error: The memory has been already allocated!");

	if (this->m_size == 0)
		throw std::runtime_error("GPUArrayLink<T>::allocate error: Cannot allocate zero bytes of memory!");

	this->m_cuda_last_error = cudaSetDevice(this->m_device_number);
	if (this->m_cuda_last_error != cudaSuccess)
		return this->m_cuda_last_error;

	this->m_cuda_last_error = cudaMalloc(&this->m_device_data, this->m_size * sizeof(T));
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::free()
{
	this->m_cuda_last_error = cudaSetDevice(this->m_device_number);
	if (this->m_cuda_last_error != cudaSuccess)
		return this->m_cuda_last_error;

	this->m_cuda_last_error = cudaFree(this->m_device_data);
	if (this->m_cuda_last_error == cudaSuccess)
		this->m_device_data = nullptr;
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, cudaStream_t t_cuda_stream)
{
	this->m_cuda_last_error = this->copy_async(t_host_link, t_device_start_index, t_host_start_index, t_size, this->m_device_number, t_cuda_stream, cudaMemcpyHostToDevice);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host_async(HostArrayLink<T> &t_host_link, cudaStream_t t_cuda_stream)
{
	this->m_cuda_last_error = this->copy_from_host_async(t_host_link, 0, 0, t_host_link.size(), t_cuda_stream);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, cudaStream_t t_cuda_stream)
{
	this->m_cuda_last_error = this->copy_async(t_host_link, t_device_start_index, t_host_start_index, t_size, this->m_device_number, t_cuda_stream, cudaMemcpyDeviceToHost);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host_async(HostArrayLink<T>& t_host_link, cudaStream_t t_cuda_stream)
{
	return this->copy_to_host_async(t_host_link, 0, 0, this->size(), t_cuda_stream);
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size)
{
	this->m_cuda_last_error = this->copy(t_host_link, t_device_start_index, t_host_start_index, t_size, this->m_device_number, cudaMemcpyHostToDevice);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host(HostArrayLink<T> &t_host_link)
{
	this->m_cuda_last_error = this->copy_from_host(t_host_link, 0, 0, t_host_link.size());
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size)
{
	this->m_cuda_last_error = this->copy(t_host_link, t_device_start_index, t_host_start_index, t_size, this->m_device_number, cudaMemcpyDeviceToHost);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_async(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, unsigned int t_device_number, cudaStream_t t_cuda_stream, cudaMemcpyKind t_kind)
{
	auto trace_error = [this, &t_host_link, t_device_start_index, t_host_start_index, t_size, t_device_number, t_kind](std::string message) -> std::string
	{
		std::ostringstream os;
		os << "GPUArray<T>::copy_async error: " << message << " "
			<< "Device array size: " << this->m_size << " elements. "
			<< "Requested size: " << t_size << " elements. "
			<< "Requested device start index: " << t_device_start_index << ". "
			<< "Requested host start index: " << t_host_start_index << ". "
			<< "Host size: " << t_host_link.size() << " elements."
			<< "cudaMemcpyKind: " << t_kind << std::endl;
		return os.str();
	};

	if (t_device_start_index + t_size > this->m_size)
	{
		auto error_message = trace_error("Device start index is not correct!");
		throw std::runtime_error(error_message.data());
	}

	if (t_host_start_index + t_size > t_host_link.size())
	{
		auto error_message = trace_error("Host array address is not correct!");
		throw std::runtime_error(&error_message[0]);
	}

	if (this->m_device_data == nullptr)
	{
		auto error_message = trace_error("GPU pointer cannot be null!");
		throw std::runtime_error(&error_message[0]);
	}

	this->m_cuda_last_error = cudaSetDevice(t_device_number);
	if (this->m_cuda_last_error != cudaSuccess)
		return this->m_cuda_last_error;

	if (t_kind == cudaMemcpyDeviceToHost)
		this->m_cuda_last_error = cudaMemcpyAsync(t_host_link[t_host_start_index], (*this)[t_device_start_index],
			t_size * sizeof(T), t_kind, t_cuda_stream);
	else if (t_kind == cudaMemcpyHostToDevice)
		this->m_cuda_last_error = cudaMemcpyAsync((*this)[t_device_start_index], t_host_link[t_host_start_index],
			t_size * sizeof(T), t_kind, t_cuda_stream);
	else throw std::runtime_error("GPUArray<T>::copy_async error: Operation not supported!");

	return this->m_cuda_last_error;
}


template<typename T>
cudaError_t DeviceArrayLink<T>::copy_async(DeviceArrayLink<T> &t_device_link, std::size_t t_src_start_index, std::size_t t_dst_start_index, std::size_t t_size, cudaStream_t t_cuda_stream)
{
	auto trace_error = [this, &t_device_link, t_src_start_index, t_dst_start_index, t_size](std::string message) -> std::string
	{
		std::ostringstream os;
		os << "GPUArray<T>::copy_async error: " << message << " "
			<< "Source array size: " << this->m_size << " elements. "
			<< "Requested size: " << t_size << " elements. "
			<< "Requested source start index: " << t_src_start_index << ". "
			<< "Requested destination start index: " << t_dst_start_index << ". "
			<< "Destination array size: " << t_device_link.size() << " elements." << std::endl;
		return os.str();
	};

	if (t_src_start_index + t_size > this->m_size)
	{
		auto error_message = trace_error("Source start index is not correct!");
		throw std::runtime_error(error_message.data());
	}

	if (t_dst_start_index + t_size > t_device_link.size())
	{
		auto error_message = trace_error("Destination array address is not correct!");
		throw std::runtime_error(error_message.data());
	}

	if (this->m_device_data == nullptr)
	{
		auto error_message = trace_error("GPU pointer cannot be null!");
		throw std::runtime_error(error_message.data());
	}

	if (this->m_device_number != t_device_link.device_number())
	{
		auto error_message = trace_error("Data transfer between different devices is not supported!");
		throw std::runtime_error(error_message.data());
	}

	this->m_cuda_last_error = cudaSetDevice(this->m_device_number);
	if (this->m_cuda_last_error != cudaSuccess)
		return this->m_cuda_last_error;

	this->m_cuda_last_error = cudaMemcpyAsync(t_device_link[t_dst_start_index], (*this)[t_src_start_index], 
			t_size * sizeof(T), cudaMemcpyDeviceToDevice, t_cuda_stream);

	return this->m_cuda_last_error;
}


template<typename T>
cudaError_t DeviceArrayLink<T>::copy(HostArrayLink<T> &t_host_link, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size, unsigned int t_device_number, cudaMemcpyKind t_kind)
{
	auto trace_error = [this, &t_host_link, t_device_start_index, t_host_start_index, t_size, t_device_number, t_kind](std::string message) -> std::string
	{
		std::ostringstream os;
		os << "GPUArray<T>::copy error: " << message << " "
			<< "Device array size: " << this->m_size << " elements. "
			<< "Requested size: " << t_size << " elements. "
			<< "Requested device start index: " << t_device_start_index << ". "
			<< "Requested host start index: " << t_host_start_index << ". "
			<< "Host size: " << t_host_link.size() << " elements."
			<< "cudaMemcpyKind: " << t_kind << std::endl;
		return os.str();
	};

	if (t_device_start_index + t_size > this->m_size)
	{
		auto error_message = trace_error("Device start index is not correct!");
		throw std::runtime_error(error_message.data());
	}

	if (t_host_start_index + t_size > t_host_link.size())
	{
		auto error_message = trace_error("Host array address is not correct!");
		throw std::runtime_error(&error_message[0]);
	}

	if (this->m_device_data == nullptr)
	{
		auto error_message = trace_error("GPU pointer cannot be null!");
		throw std::runtime_error(&error_message[0]);
	}

	this->m_cuda_last_error = cudaSetDevice(t_device_number);
	if (this->m_cuda_last_error != cudaSuccess)
		return this->m_cuda_last_error;

	if (t_kind == cudaMemcpyDeviceToHost)
		this->m_cuda_last_error = cudaMemcpy(t_host_link[t_host_start_index], (*this)[t_device_start_index],
			t_size * sizeof(T), t_kind);
	else if (t_kind == cudaMemcpyHostToDevice)
		this->m_cuda_last_error = cudaMemcpy((*this)[t_device_start_index], t_host_link[t_host_start_index],
			t_size * sizeof(T), t_kind);
	else throw std::runtime_error("GPUArray<T>::copy error: Operation not supported!");

	return this->m_cuda_last_error;
}



template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host(T* t_data, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size)
{
	return this->copy_from_host(HostArrayLink<T>{t_data, t_size}, t_device_start_index, t_host_start_index, t_size);
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host(T* t_data, std::size_t t_size)
{
	return this->copy_from_host(HostArrayLink<T>{t_data, t_size});
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host(T* t_data, std::size_t t_device_start_index, std::size_t t_host_start_index, std::size_t t_size)
{
	return this->copy_to_host(HostArrayLink<T>{t_data, t_size}, t_device_start_index, t_host_start_index, t_size);
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host(T* t_data, std::size_t t_size)
{
	HostArrayLink<T> link{ t_data, t_size };
	return this->copy_to_host(link, std::size_t(0), std::size_t(0), t_size);
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_from_host(std::vector<T>& t_host_data)
{
	return this->copy_from_host(t_host_data.data(), t_host_data.size());
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_host(std::vector<T>& t_host_data)
{
	return this->copy_to_host(t_host_data.data(), t_host_data.size());
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_device(DeviceArrayLink<T>& t_device_data)
{
	if (t_device_data.device_number() != this->m_device_number)
		throw std::runtime_error("GPUArray<T>::copy_to_device error: Destination device must be the same as source device!");

	this->m_cuda_last_error = cudaMemcpy(t_device_data.data(), this->m_device_data,
		m_size * sizeof(T), cudaMemcpyDeviceToDevice);
	return this->m_cuda_last_error;
}

template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_device_async(DeviceArrayLink<T>& t_device_data, cudaStream_t t_cuda_stream)
{
	if (t_device_data.device_number() != this->m_device_number)
		throw std::runtime_error("GPUArray<T>::copy_to_device_async error: Destination device must be the same as source device!");

	this->m_cuda_last_error = cudaMemcpyAsync(t_device_data.data(), this->m_device_data,
		m_size * sizeof(T), cudaMemcpyDeviceToDevice, t_cuda_stream);
	return this->m_cuda_last_error;
}


template<typename T>
cudaError_t DeviceArrayLink<T>::copy_to_device_async(DeviceArrayLink<T>& t_device_data, std::size_t t_src_start_index, std::size_t t_dst_start_index, std::size_t t_size, cudaStream_t cuda_stream)
{
	return this->copy_async(t_device_data, t_src_start_index, t_dst_start_index, t_size, cuda_stream);
}

template<typename T>
cudaError_t DeviceArrayLink<T>::cuda_last_error()
{
	return cudaError_t(this->m_cuda_last_error);
}


