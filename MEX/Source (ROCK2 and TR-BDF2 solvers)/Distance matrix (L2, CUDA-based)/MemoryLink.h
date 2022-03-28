#pragma once

#include <stdexcept>
#include <vector>

template <typename T> class MemoryLink
{
public:
	MemoryLink(T* t_data, size_t t_size);
	MemoryLink(std::vector<T>& t_data);
	size_t size() const;
	T* data() const;
	T& operator [] (const size_t t_index) const;
	T* begin();
	T* end();
	const T* begin() const;
	const T* end() const;
private:
	T* m_data = nullptr;
	T* m_end = nullptr;
	size_t m_size = 0;
};

template<typename T> inline MemoryLink<T>::MemoryLink(T * t_data, size_t t_size)
{
	this->m_data = t_data;
	this->m_size = t_size;
	this->m_end = t_data + t_size;
}

template<typename T> inline MemoryLink<T>::MemoryLink(std::vector<T>& t_data)
	: MemoryLink<T>(t_data.data(), t_data.size()) {}

template<typename T> inline size_t MemoryLink<T>::size() const
{
	return this->m_size;
}

template<typename T> inline T* MemoryLink<T>::data() const
{
	return this->m_data;
}

template<typename T> inline T & MemoryLink<T>::operator[](const size_t t_index) const
{
	if (t_index >= this->m_size)
		throw std::out_of_range("The index was out of range!");
	return this->m_data + t_index;
}

template<typename T> inline T* MemoryLink<T>::begin()
{
	return this->m_data;
}

template<typename T> inline T* MemoryLink<T>::end()
{
	return this->m_end;
}

template<typename T> inline const T* MemoryLink<T>::begin() const 
{
	return this->m_data;
}

template<typename T> inline const T* MemoryLink<T>::end() const
{
	return this->m_end;
}
