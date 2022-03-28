#pragma once

#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

#include <stdexcept>

#include "MemoryLink.h"
#include "RunResult.cuh"

cublasStatus_t computeMatrixNorm(
	cublasHandle_t t_handle,
	size_t t_dimension,
	size_t t_column_number,
	float* t_dev_matrix_squared,
	float* t_dev_one_vector,
	float* t_dev_matrix_norm
)
{
	float alpha = 1;
	float beta = 0;

	return cublasSgemv(
		t_handle,
		CUBLAS_OP_T,
		static_cast<int>(t_dimension),
		static_cast<int>(t_column_number),
		&alpha,
		t_dev_matrix_squared,
		static_cast<int>(t_dimension),
		t_dev_one_vector,
		1,
		&beta,
		t_dev_matrix_norm,
		1
	);
}


cublasStatus_t computeMatrixNorm(
	cublasHandle_t t_handle,
	size_t t_dimension,
	size_t t_column_number,
	double* t_dev_matrix_squared,
	double* t_dev_one_vector,
	double* t_dev_matrix_norm
)
{
	double alpha = 1;
	double beta = 0;

	return cublasDgemv(
		t_handle,
		CUBLAS_OP_T,
		static_cast<int>(t_dimension),
		static_cast<int>(t_column_number),
		&alpha,
		t_dev_matrix_squared,
		static_cast<int>(t_dimension),
		t_dev_one_vector,
		1,
		&beta,
		t_dev_matrix_norm,
		1
	);
}


cublasStatus_t computeDotProduct(
	cublasHandle_t t_handle, 
	size_t t_first_column_number,
	size_t t_second_column_number,
	size_t t_dimension,
	double* t_dev_first_matrix,
	double* t_dev_second_matrix,
	double* t_dot_product)
{
	double alpha = 1;
	double beta = 0;

	return cublasDgemm(
		t_handle,
		CUBLAS_OP_T, // operation with first matrix (transpose)
		CUBLAS_OP_N, // operation with second matrix (none)
		static_cast<int>(t_first_column_number), // number of rows in the first matrix (transposed)
		static_cast<int>(t_second_column_number), // number of columns in the second matrix
		static_cast<int>(t_dimension), // number of columns in the first matrix (transposed) and also number of rows in the second matrix
		&alpha,
		t_dev_first_matrix,
		static_cast<int>(t_dimension), // leading dimension in the first matrix (non-transposed)
		t_dev_second_matrix,
		static_cast<int>(t_dimension), // leading dimension in the second matrix (non-transposed)
		&beta,
		t_dot_product,
		static_cast<int>(t_first_column_number) // leading dimension in the result matrix (non-transposed)
	);
}

cublasStatus_t computeDotProduct(
	cublasHandle_t t_handle,
	size_t t_first_column_number,
	size_t t_second_column_number,
	size_t t_dimension,
	float* t_dev_first_matrix,
	float* t_dev_second_matrix,
	float* t_dot_product)
{
	float alpha = 1;
	float beta = 0;

	return cublasSgemm(
		t_handle,
		CUBLAS_OP_T, // operation with first matrix (transpose)
		CUBLAS_OP_N, // operation with second matrix (none)
		static_cast<int>(t_first_column_number), // number of rows in the first matrix (transposed)
		static_cast<int>(t_second_column_number), // number of columns in the second matrix
		static_cast<int>(t_dimension), // number of columns in the first matrix (transposed) and also number of rows in the second matrix
		&alpha,
		t_dev_first_matrix,
		static_cast<int>(t_dimension), // leading dimension in the first matrix (non-transposed)
		t_dev_second_matrix,
		static_cast<int>(t_dimension), // leading dimension in the second matrix (non-transposed)
		&beta,
		t_dot_product,
		static_cast<int>(t_first_column_number) // leading dimension in the result matrix (non-transposed)
	);
}


template <typename value_type> struct pow2_operation
{
	__host__ __device__ value_type operator()(const value_type &x) const
	{
		return x * x;
	}
};

template <typename value_type> __global__ void compute_distance_matrix(
	const value_type* __restrict__ dev_first_matrix_norm,
	const value_type* __restrict__ dev_second_matrix_norm,
	const value_type* __restrict__ dev_dot_product,
	value_type* __restrict__ dev_distance_matrix,
	const size_t first_column_number,
	const size_t second_column_number
)
{

	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	const int j = threadIdx.y + blockIdx.y * blockDim.y;

	if ((i < second_column_number) && (j < first_column_number))
		dev_distance_matrix[i * first_column_number + j] =
		abs(dev_first_matrix_norm[j] + dev_second_matrix_norm[i] - 2 * dev_dot_product[i * first_column_number + j]);

}


template <typename value_type> RunResult runDeviceComputations(
	const MemoryLink<value_type>& t_first_matrix,
	const MemoryLink<value_type>& t_second_matrix,
	MemoryLink<value_type>& t_distance_matrix,
	const size_t t_dimension)
{
	size_t first_column_number = t_first_matrix.size() / t_dimension;
	size_t second_column_number = t_second_matrix.size() / t_dimension;

	thrust::device_vector<value_type> dev_first_matrix(t_first_matrix.begin(), t_first_matrix.end());
	thrust::device_vector<value_type> dev_second_matrix(t_second_matrix.begin(), t_second_matrix.end());

	// creating CUBLAS handle
	cublasHandle_t handle;
	CUBLAS_CALL(cublasCreate(&handle), "CUBLAS handle creation failed!");

	// we will compute the distance matrix in the form ||x-y||^2 = ||x||^2 + ||y||^2 - 2(x,y)

	// computing ||x||^2 and ||y||^2
	thrust::device_vector<value_type> dev_first_matrix_norm(first_column_number);
	thrust::device_vector<value_type> dev_second_matrix_norm(second_column_number);

	// first we square the matricex
	thrust::device_vector<value_type> dev_first_matrix_squared(t_first_matrix.size());
	thrust::device_vector<value_type> dev_second_matrix_squared(t_second_matrix.size());

	thrust::transform(dev_first_matrix.begin(), dev_first_matrix.end(), dev_first_matrix_squared.begin(), pow2_operation<value_type>());
	thrust::transform(dev_second_matrix.begin(), dev_second_matrix.end(), dev_second_matrix_squared.begin(), pow2_operation<value_type>());

	// next we compute the norms
	// empty vectors, needed for calling cublasSgemv function (matrix multiplication)
	thrust::device_vector<value_type> dev_first_one_vector(t_dimension, value_type(1));
	thrust::device_vector<value_type> dev_second_one_vector(t_dimension, value_type(1));

	// inside computeMatrixNorm() we are calling CUBLAS matrix-vector multiplication routine (res = alpha * A^t * b + beta * c)

	// dev_first_matrix in (dim, N1) -> A^t in (N1, dim)
	// dev_first_one_vector in (dim, 1)
	// dev_first_matrix_norm (result) in (N1, 1)
	CUBLAS_CALL(computeMatrixNorm(
		handle,
		t_dimension,
		first_column_number,
		thrust::raw_pointer_cast(dev_first_matrix_squared.data()),
		thrust::raw_pointer_cast(dev_first_one_vector.data()),
		thrust::raw_pointer_cast(dev_first_matrix_norm.data())
	), "CUBLAS call failed (computing first matrix norm)!");

	// dev_second_matrix in (dim, N2) -> A^t in (N2, dim)
	// dev_second_one_vector in (dim, 1)
	// dev_second_matrix_norm (result) in (N2, 1)
	CUBLAS_CALL(computeMatrixNorm(
		handle,
		t_dimension,
		second_column_number,
		thrust::raw_pointer_cast(dev_second_matrix_squared.data()),
		thrust::raw_pointer_cast(dev_second_one_vector.data()),
		thrust::raw_pointer_cast(dev_second_matrix_norm.data())
	), "CUBLAS call failed (computing second matrix norm)!");	

	// calculating the dot product
	thrust::device_vector<value_type> dev_dot_product(first_column_number * second_column_number);

	// inside computeDotProduct() we are calling CUBLAS matrix-matrix multiplication routine (C = alpha * A^t * B^t + beta * C
	// dev_first_matrix (dim, N1) -> A^t in (N1, dim)
	// dev_second_matrix (dim, N2)
	// dev_dot_product (result) in (N1, N2) 
	CUBLAS_CALL(computeDotProduct(
		handle,
		first_column_number,
		second_column_number,
		t_dimension,
		thrust::raw_pointer_cast(dev_first_matrix.data()),
		thrust::raw_pointer_cast(dev_second_matrix.data()),
		thrust::raw_pointer_cast(dev_dot_product.data())
	), "CUBLAS call failed (computing dot product)!");

	// calculating the distance matrix
	thrust::device_vector<value_type> dev_distance_matrix(first_column_number * second_column_number);

	unsigned int blockDimX = 32;
	unsigned int blockDimY = 32;

	dim3 block{ blockDimX, blockDimY, 1 };
	dim3 grid{ static_cast<unsigned int>(second_column_number) / blockDimY + 1, static_cast<unsigned int>(first_column_number) / blockDimX + 1 };

	compute_distance_matrix << <grid, block >> > (
		thrust::raw_pointer_cast(dev_first_matrix_norm.data()),
		thrust::raw_pointer_cast(dev_second_matrix_norm.data()),
		thrust::raw_pointer_cast(dev_dot_product.data()),
		thrust::raw_pointer_cast(dev_distance_matrix.data()),
		first_column_number,
		second_column_number
		);

	thrust::copy(dev_distance_matrix.begin(), dev_distance_matrix.end(), t_distance_matrix.begin());

	return { CUBLAS_STATUS_SUCCESS, "" };

}