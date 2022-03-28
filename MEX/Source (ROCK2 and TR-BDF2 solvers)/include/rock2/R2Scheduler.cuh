#pragma once
// This abstract class defines the structure of communication between R2SimulationRunner and the implementation of R2Handler.

#include <vector>

#include "rock2/R2OperationQueueManager.cuh"

#include <string>

template <typename value_type, typename index_type>
class R2Scheduler
{
protected:
	index_type m_sim_number;

	R2OperationQueueManager<index_type> m_operation_queue;
	std::vector<value_type> m_time_step_queue;
	
public:
	R2Scheduler(index_type t_sim_number);
	index_type get_sim_number();
	void clear_queues();
	// yprev = y;
	void schedule_prev_state_overwrite_stage(
		index_type t_runner_number,
		index_type mdeg
	); // OP-0
	// yjm2 = yprev; yjm1 = yprev + dt * recf(mr) * f(y_prev); if mdeg < 2 y = yjm1;
	void schedule_r2_initial_stage(
		index_type t_runner_number,
		value_type t_new_time_step,
		index_type mr,
		index_type mdeg
	); // OP-1
	// y=temp1*f(yjm1)+temp2*yjm1+temp3*yjm2;
	void schedule_r2_rec_stage(
		index_type t_runner_number,
		index_type mr,
		index_type i
	); // OP-2
	// yjm1=y+temp1*f(neqn,y); y = yjm1 + temp1*f(neqn,yjm1) + temp2*(y-yjm2);
	// err = sum((temp2*(y-yjm2)./(ato+t_ci1)).^2); err=sqrt(err/neqn);
	index_type schedule_r2_finishing_stage(
		index_type t_runner_number,
		index_type mz
	); // OP-3
	// eigmax = rho(yprev)
	index_type schedule_spectral_radius_est_stage(
		index_type t_runner_number
	); // OP-4
	// norm = max_norm(yprev)
	index_type schedule_rhs_norm_stage(
		index_type t_runner_number
	); // OP-5
	virtual value_type get_local_error(index_type t_queue_number) = 0;
	virtual value_type get_spectral_radius(index_type t_queue_number) = 0;
	virtual value_type get_rhs_norm(index_type t_queue_number) = 0;

	virtual void log_sys_state(std::string t_file_name) = 0;
};

template <typename value_type, typename index_type>
R2Scheduler<value_type, index_type>::R2Scheduler(index_type t_sim_number):
	m_sim_number(t_sim_number), m_time_step_queue(t_sim_number), m_operation_queue(t_sim_number)
{
	this->clear_queues();
}

template <typename value_type, typename index_type>
index_type R2Scheduler<value_type, index_type>::get_sim_number()
{
	return m_sim_number;
}

template<typename value_type, typename index_type>
void R2Scheduler<value_type, index_type>::clear_queues()
{
	m_operation_queue.clear();
}

template<typename value_type, typename index_type>
void R2Scheduler<value_type, index_type>::schedule_prev_state_overwrite_stage(index_type t_runner_number, index_type mdeg)
{
	m_operation_queue.insert(R2OperationType::PreviousStepOverwrite, t_runner_number, mdeg);
}

template<typename value_type, typename index_type>
void R2Scheduler<value_type, index_type>::schedule_r2_initial_stage(index_type t_runner_number, value_type t_new_time_step, index_type mr, index_type mdeg)
{
	auto counter = m_operation_queue.get_operation_queue_length(R2OperationType::InitialStage);
	m_time_step_queue[counter] = t_new_time_step;
	m_operation_queue.insert(R2OperationType::InitialStage, t_runner_number, mr, mdeg );
}

template<typename value_type, typename index_type>
void R2Scheduler<value_type, index_type>::schedule_r2_rec_stage(index_type t_runner_number, index_type mr, index_type i)
{
	m_operation_queue.insert(R2OperationType::RecursiveStage, t_runner_number, mr, i );
}

template<typename value_type, typename index_type>
index_type R2Scheduler<value_type, index_type>::schedule_r2_finishing_stage(index_type t_runner_number, index_type mz)
{
	auto queue_number = m_operation_queue.get_operation_queue_length(R2OperationType::FinishingProcedure);
	m_operation_queue.insert(R2OperationType::FinishingProcedure, t_runner_number, mz);

	return queue_number;
}

template<typename value_type, typename index_type>
index_type R2Scheduler<value_type, index_type>::schedule_spectral_radius_est_stage(index_type t_runner_number)
{
	auto queue_number = m_operation_queue.get_operation_queue_length(R2OperationType::SpectralRadiusEstimation);
	m_operation_queue.insert(R2OperationType::SpectralRadiusEstimation, t_runner_number);
	return queue_number;
}

template<typename value_type, typename index_type>
index_type R2Scheduler<value_type, index_type>::schedule_rhs_norm_stage(index_type t_runner_number)
{
	return 0;
}

