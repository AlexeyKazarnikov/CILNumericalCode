#pragma once

#include <algorithm>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

void read_vector(const std::string& filename, std::vector<double>& v)
{
	double val;
	std::ifstream istream(filename, std::ios::in);
	if (!istream.is_open())
	{
		std::cout << "Failed to open file '" << filename << "'!" << std::endl;
		throw std::runtime_error("Failed to open file!");
		//return;
	}

	v.clear();

	while (istream >> val)
	{
		v.push_back(val);
	}
	istream.close();
};

template <typename value_type>
void write_vector(const std::string& filename, std::vector<value_type>& v)
{
	std::ofstream ostream(filename, std::ios::trunc);
	if (!ostream.is_open())
	{
		std::cout << "Failed to open file '" << filename << "'!" << std::endl;
		throw std::runtime_error("Failed to open file!");
		//return;
	}

	ostream << std::setprecision(32);

	for (auto k = 0; k < v.size() - 1; ++k)
		ostream << v[k] << '\t';

	ostream << v[v.size() - 1];

	ostream.close();
}

template <typename T> T max_norm(const std::vector<T>& a, const std::vector<T>& b)
{
	T result = 0;
	if (a.size() != b.size())
		throw std::exception("vectors must have the same length!");

	for (unsigned int k = 0; k < a.size(); ++k)
		result = std::max(result, abs(a[k] - b[k]));

	return result;
}