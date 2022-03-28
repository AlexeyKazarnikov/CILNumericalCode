#pragma once
// This class provides a base for any handler, capable of executing ROCK2 operations. 
// It is assumed that computations are handled by handle_computations() method.

#include <vector>
#include <stdexcept>

#include "rock2/R2DataArrays.cuh"

template <typename value_type, typename index_type>
class R2Handler
{
protected:
	index_type m_sim_number;
	index_type m_sys_size;
	std::vector<value_type> m_model_parameters;
	value_type m_atol;
	value_type m_rtol;

	std::vector<value_type> m_fp1;
	std::vector<value_type> m_fp2;
	std::vector<value_type> m_recf;

public:
	R2Handler(
		index_type t_sim_number,
		index_type t_sys_size,
		const std::vector<value_type>& t_model_parameters,
		value_type t_atol = 1e-3,
		value_type t_rtol = 1e-3
	)
		: 
		m_sim_number(t_sim_number),
		m_sys_size(t_sys_size),
		m_model_parameters(t_model_parameters),
		m_atol(t_atol),
		m_rtol(t_rtol)
	{
		init_r2_data_arrays(m_fp1, m_fp2, m_recf);
	}

	virtual void handle_computations()
	{
		throw std::logic_error("Cannot handle computations without defining exact implementation of the method!");
	}
};