#pragma once

#include <algorithm>
#include <map>
#include <numeric>
#include <unordered_map>
#include <vector>


enum R2OperationType
{
	PreviousStepOverwrite = 0,
	InitialStage = 1,
	RecursiveStage = 2,
	FinishingProcedure = 3,
	SpectralRadiusEstimation = 4,
	RhsNormEstimation = 5
};

template <typename index_type>
class R2OperationQueueManager
{
private:
	std::vector<index_type> m_operation_queue;
	std::unordered_map<R2OperationType, index_type> m_operation_queue_data;
	const std::vector<size_t> m_item_sizes = { 2, 3, 3, 2, 1, 1 };
	std::vector<size_t> m_start_indexes;
	const size_t m_reserved_size;

public:
	R2OperationQueueManager(size_t t_reserved_size);
	size_t get_operation_queue_start(R2OperationType t_operation);
	size_t get_operation_queue_end(R2OperationType t_operation);
	size_t get_operation_queue_length(R2OperationType t_operation);

	index_type& operator[](std::size_t t_index);
	const index_type& operator[](std::size_t t_index) const;

	void insert(R2OperationType t_operation, index_type t_data);
	void insert(R2OperationType t_operation, index_type t_data_1, index_type t_data_2);
	void insert(R2OperationType t_operation, index_type t_data_1, index_type t_data_2, index_type t_data_3);

	index_type* data();
	size_t size();
	void clear();
};

template<typename index_type>
R2OperationQueueManager<index_type>::R2OperationQueueManager(size_t t_reserved_size)
	: m_reserved_size(t_reserved_size)
{
	size_t queue_total_size = t_reserved_size * std::accumulate(m_item_sizes.begin(), m_item_sizes.end(), 0);
	m_operation_queue.resize(queue_total_size);
	this->clear();

	m_start_indexes.resize(m_item_sizes.size());
	for (int k = R2OperationType::PreviousStepOverwrite; k < R2OperationType::RhsNormEstimation; ++k)
	{
		size_t operation_index = k;
		size_t offset = 0;
		for (unsigned int k1 = 0; k1 < operation_index; ++k1)
		{
			offset += m_item_sizes[k1] * m_reserved_size;
		}
		m_start_indexes[k] = offset;
	}
}

template<typename index_type>
size_t R2OperationQueueManager<index_type>::get_operation_queue_start(R2OperationType t_operation)
{
	size_t operation_index = static_cast<size_t>(t_operation);
	return m_start_indexes[operation_index];
}

template<typename index_type>
size_t R2OperationQueueManager<index_type>::get_operation_queue_end(R2OperationType t_operation)
{
	size_t operation_index = static_cast<size_t>(t_operation);
	return this->get_operation_queue_start(t_operation) + m_item_sizes[operation_index] * m_operation_queue_data[t_operation];
}

template<typename index_type>
size_t R2OperationQueueManager<index_type>::get_operation_queue_length(R2OperationType t_operation)
{
	return m_operation_queue_data[t_operation];
}

template<typename index_type>
index_type& R2OperationQueueManager<index_type>::operator[](std::size_t t_index)
{
	return m_operation_queue[t_index];
}

template<typename index_type>
const index_type& R2OperationQueueManager<index_type>::operator[](std::size_t t_index) const
{
	return m_operation_queue[t_index];
}

template<typename index_type>
void R2OperationQueueManager<index_type>::insert(R2OperationType t_operation, index_type t_data)
{
	if (m_item_sizes[t_operation] != 1)
		throw std::exception();
	auto offset = this->get_operation_queue_end(t_operation);
		m_operation_queue[offset] = t_data;
	++m_operation_queue_data[t_operation];
}

template<typename index_type>
void R2OperationQueueManager<index_type>::insert(R2OperationType t_operation, index_type t_data_1, index_type t_data_2)
{
	if (m_item_sizes[t_operation] != 2)
		throw std::exception();
	auto offset = this->get_operation_queue_end(t_operation);
	m_operation_queue[offset] = t_data_1;
	m_operation_queue[offset + 1] = t_data_2;
	++m_operation_queue_data[t_operation];
}

template<typename index_type>
void R2OperationQueueManager<index_type>::insert(R2OperationType t_operation, index_type t_data_1, index_type t_data_2, index_type t_data_3)
{
	if (m_item_sizes[t_operation] != 3)
		throw std::exception();
	auto offset = this->get_operation_queue_end(t_operation);
	m_operation_queue[offset] = t_data_1;
	m_operation_queue[offset + 1] = t_data_2;
	m_operation_queue[offset + 2] = t_data_3;
	++m_operation_queue_data[t_operation];
}

template<typename index_type>
index_type* R2OperationQueueManager<index_type>::data()
{
	return m_operation_queue.data();
}

template<typename index_type>
size_t R2OperationQueueManager<index_type>::size()
{
	return m_operation_queue.size();
}

template<typename index_type>
void R2OperationQueueManager<index_type>::clear()
{
	for (int k = R2OperationType::PreviousStepOverwrite; k < R2OperationType::RhsNormEstimation; ++k)
		m_operation_queue_data[static_cast<R2OperationType>(k)] = 0;
}
