#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename value_type, typename index_type> class shift_handler
{
private:
	const value_type* m_data;
	const index_type m_shift_index;
	const value_type m_shift_value;
public:
	__device__ shift_handler(const value_type* t_data, const index_type t_shift_index, const value_type t_shift_value)
		: m_data(t_data), m_shift_index(t_shift_index), m_shift_value(t_shift_value) {}
	__device__ value_type operator[](const index_type t_index) const
	{
		return m_data[t_index] + m_shift_value * (t_index == m_shift_index);
	}
	__device__ const value_type* data() const
	{
		return m_data;
	}
};