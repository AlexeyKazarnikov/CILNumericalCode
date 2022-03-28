#pragma once
// This class handles the process of executing the ROCK2 simulation. 
// The idea is that no computations are performed here, only the operations are requested and the results are retrieved.

#include <iostream>
#include<vector>

#include "R2Scheduler.cuh"

template <typename value_type, typename index_type>
struct R2RunnerStatistics
{
	index_type accepted_step_number = 0;
	value_type min_spectral_radius = 0;
	value_type max_spectral_radius = 0;
	index_type max_stage_number = 0;
	index_type rejected_step_number = 0;
	index_type rhs_evaluation_number = 0;
	index_type step_number = 0;
};

template <typename value_type, typename index_type> class R2Runner
{
private:
	index_type m_runner_number;
	index_type m_stage = 1;
	index_type m_return_code = 0;

	bool m_is_rejected = false;
	bool m_is_last_step = false;

	index_type m_spectral_radius_queue_number = 0;
	index_type m_local_error_queue_number = 0;
	index_type m_rhs_norm_queue_number = 0;

	value_type m_time_step;
	value_type m_final_time_point;
	value_type m_conv_norm;
	index_type m_conv_stage_number;
	value_type m_u_round;

	index_type m_degree = 0;
	index_type m_degree_prev = 0;
	value_type m_local_error = 0;
	value_type m_local_error_prev = 0;
	value_type m_max_time_step_increase_factor = value_type(2);
	index_type m_pos_fp = 0;
	index_type m_pos_recf = 0;
	value_type m_prev_rejection_time = 0;
	index_type m_rejected_step_number = 0;
	value_type m_rhs_norm = 0;
	value_type m_spectral_radius = 0;
	index_type m_spectral_radius_stage = 0;
	index_type m_stage_counter = 0;
	value_type m_time = 0;
	value_type m_time_step_prev = 0;
	index_type m_max_step_number = 0;

	R2RunnerStatistics<value_type, index_type> m_runner_statistics;

	std::vector<index_type> m_ms = {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	17, 18, 19, 20, 22, 24, 26, 28, 30, 33, 36, 39, 43,
	47, 51, 56, 61, 66, 72, 78, 85, 93, 102, 112, 123,
	135, 148, 163, 180, 198
	};
public:
	R2Runner(
		index_type t_runner_number,
		value_type t_time_step = 1e-5,
		value_type t_final_time_point = 100,
		value_type t_conv_norm = 1e-4,
		index_type t_conv_stage_number = 150,
		index_type t_max_step_number = 500,
		value_type t_u_round = 1e-16
		)
		: 
		m_runner_number(t_runner_number), 
		m_time_step(t_time_step), 
		m_final_time_point(t_final_time_point),
		m_conv_norm(t_conv_norm),
		m_conv_stage_number(t_conv_stage_number),
		m_max_step_number(t_max_step_number),
		m_u_round(t_u_round)
	{}
	void run(R2Scheduler<value_type, index_type>& scheduler);
	bool is_active();
	bool is_convergence_reached();
};

template<typename value_type, typename index_type>
void R2Runner<value_type, index_type>::run(R2Scheduler<value_type, index_type>& scheduler)
{
	bool exit_flag = false;

	while (!exit_flag)
	{
		if (m_runner_statistics.step_number >= m_max_step_number)
		{
			m_stage = 0;
			break;
		}

		switch (m_stage)
		{
		case 0: // execution is finished
			exit_flag = true;
			break;
		case 1: { //
			if (!m_is_rejected)
			{
				scheduler.schedule_prev_state_overwrite_stage(m_runner_number, m_degree_prev);
				m_local_error_prev = m_local_error;
			}
				
			if (value_type(1.1) * m_time_step >= abs(m_final_time_point - m_time))
			{
				m_time_step = abs(m_final_time_point - m_time);
				m_is_last_step = true;
			}

			if (m_time_step < value_type(10) * m_u_round)
			{
				//stderr << "Tolerances are too small!" << std::endl;
				m_return_code = 1;
				m_stage = 0;
				exit_flag = true;
			}

			if (m_spectral_radius_stage == 0)
			{
				m_spectral_radius_queue_number = scheduler.schedule_spectral_radius_est_stage(m_runner_number);

				m_stage = 2;
				exit_flag = true;
			}
			else
			{
				m_stage = 3;
			}

			break;
		}
		case 2: {
			m_spectral_radius = scheduler.get_spectral_radius(m_spectral_radius_queue_number);

			if (m_spectral_radius > m_runner_statistics.max_spectral_radius)
				m_runner_statistics.max_spectral_radius = m_spectral_radius;

			if (m_runner_statistics.step_number == 0)
				m_runner_statistics.min_spectral_radius = m_spectral_radius;

			if (m_spectral_radius < m_runner_statistics.min_spectral_radius)
				m_runner_statistics.min_spectral_radius = m_spectral_radius;

			m_stage = 3;
			break;
		}
		case 3: {
			m_degree = static_cast<index_type>(sqrt((value_type(1.5) + m_time_step * m_spectral_radius) / value_type(0.811)) + 1);

			if (m_degree > 200)
			{
				m_time_step = value_type(0.8) * (value_type(200) * value_type(200) * value_type(0.811) - value_type(1.5)) / m_spectral_radius;
				m_degree = 200;
				m_is_last_step = false;
			}

			m_degree = std::max(m_degree, index_type(3u)) - index_type(2u);

			if (m_degree != m_degree_prev)
			{
				m_pos_recf = 0;

				for (unsigned int k = 0; k < 46; ++k)
				{
					if (value_type(m_ms[k]) / value_type(m_degree) >= 1)
					{
						m_degree = m_ms[k];
						m_pos_fp = k;
						break;
					}
					m_pos_recf += m_ms[k] * 2 - 1;
				}
			}

			if (m_degree + 2 > m_runner_statistics.max_stage_number)
				m_runner_statistics.max_stage_number = m_degree + 2;

			scheduler.schedule_r2_initial_stage(m_runner_number, m_time_step, m_pos_recf, m_degree);

			if (m_degree < 2)
				m_stage = 5;
			else
			{
				m_stage_counter = 2;
				m_stage = 4;	
			}
			break;
		}
		case 4: {
			scheduler.schedule_r2_rec_stage(m_runner_number, m_pos_recf, m_stage_counter);
			if (m_stage_counter < m_degree)
			{
				m_stage_counter += 1;
				exit_flag = true;
			}
			else
			{
				m_stage = 5;
			}
			break;
		}
		case 5: {

			m_local_error_queue_number = scheduler.schedule_r2_finishing_stage(m_runner_number, m_pos_fp);
			m_stage = 6;
			exit_flag = true;
			break;
		}
		case 6: {
			m_local_error = scheduler.get_local_error(m_local_error_queue_number);

			//if (m_runner_number == 0)
			//	std::cout << m_local_error << std::endl;

			m_degree_prev = m_degree;
			m_runner_statistics.step_number+=1;
			m_runner_statistics.rhs_evaluation_number += m_degree + 2;

			if (m_runner_statistics.step_number == m_max_step_number) // maximal number of steps exceeded, saving data and exiting
			{
				scheduler.schedule_prev_state_overwrite_stage(m_runner_number, m_degree_prev);
				m_stage = 0;
				exit_flag = true;
				break;
			}

			value_type time_step_factor = sqrt(value_type(1) / m_local_error);
			if (m_local_error_prev != 0 && !m_is_rejected)
			{
				value_type time_step_factor_prev =
					sqrt(m_local_error_prev) * time_step_factor * time_step_factor * (m_time_step / m_time_step_prev);
				time_step_factor = std::min(time_step_factor, time_step_factor_prev);
			}
			if (m_is_rejected)
				m_max_time_step_increase_factor = 1;

			time_step_factor = std::min(m_max_time_step_increase_factor, std::max(value_type(0.1), value_type(0.8) * time_step_factor));
			value_type new_time_step = m_time_step * time_step_factor;

			if (m_local_error < 1)
			{
				m_runner_statistics.accepted_step_number += 1;
				m_max_time_step_increase_factor = 2;
				m_time += m_time_step;

				if (m_is_rejected) // previous step was rejected
				{
					new_time_step = std::min(new_time_step, m_time_step);
					if (m_final_time_point < m_time)
						new_time_step = std::max(new_time_step, m_time_step);
					m_is_rejected = false;
					m_rejected_step_number = 0;
				}

				m_time_step_prev = m_time_step;
				m_time_step = new_time_step;
				m_spectral_radius_stage += 1;
				m_spectral_radius_stage = (m_spectral_radius_stage + 1) % 25;

				if (m_is_last_step)
				{
					scheduler.schedule_prev_state_overwrite_stage(m_runner_number, m_degree_prev);
					m_stage = 0;
					exit_flag = true;
				}
				else
				{
					m_stage = 1;
				}		
				break;
			}
			else
			{
				m_runner_statistics.rejected_step_number += 1;
				m_is_rejected = true;
				m_is_last_step = false;
				m_time_step = value_type(0.8) * new_time_step;
				if (m_runner_statistics.step_number == 0)
					m_time_step *= value_type(0.1);
				if (m_prev_rejection_time == m_time)
				{
					m_rejected_step_number += 1;
					if (m_rejected_step_number == 10)
						m_time_step = value_type(1e-5);
					m_prev_rejection_time = m_time;
				}

				if (m_spectral_radius_stage != 0)
					m_spectral_radius_stage = 0;
				else
					m_spectral_radius_stage = 1;
				m_stage = 1;
				break;
			}
		}
		} // switch
	}
}

template<typename value_type, typename index_type>
bool R2Runner<value_type, index_type>::is_active()
{
	return (m_stage != 0);
}

template<typename value_type, typename index_type>
bool R2Runner<value_type, index_type>::is_convergence_reached()
{
	return m_runner_statistics.step_number < m_max_step_number;
}
