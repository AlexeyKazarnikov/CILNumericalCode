#include <algorithm>
#include <cmath>
#include <mex.h>

//Checks if supplied array contains numeric data represented in floating point format. Returns 'true' if so, and 'false' otherwise
bool mxIsFPData(const mxArray* pArray);


//Computes distance matrix for the given sets of vectors
template<typename T>
void computeDistanceMatrix(
	const T* p_vectors1,
	size_t num_vectors1,
	const T* p_vectors2,
	size_t num_vectors2,
	unsigned int space_power,
	size_t vector_length,
	T* distance_matrix)
{
#pragma omp parallel for
	for (long long j = 0; j < static_cast<long long>(num_vectors2); ++j)
		for (long long i = 0; i < static_cast<long long>(num_vectors1); ++i)
		{
			T S = 0;
			for (size_t k = 0; k < vector_length; ++k)
			{
				T aux = std::abs(p_vectors1[vector_length * i + k] - p_vectors2[vector_length * j + k]);
				if (space_power > 0)
				{
					for (unsigned int p = 1; p < space_power; ++p)
						aux *= aux;
					S += aux;
				}
				else
				{
					S = std::max(S, aux);
				}
			}

			distance_matrix[num_vectors1 * j + i] = S;
		}
}

template<typename T>
void computeDistanceMatrix(
	const T* p_vectors,
	size_t num_vectors,
	unsigned int space_power,
	size_t vector_length,
	T* distance_matrix)
{
#pragma omp parallel for
	for (long long i = 0; i < static_cast<long long>(num_vectors); ++i)
		for (long long j = 0; j < static_cast<long long>(i); ++j)
		{
			T S = 0;
			for (size_t k = 0; k < vector_length; ++k)
			{
				T aux = std::abs(p_vectors[vector_length * i + k] - p_vectors[vector_length * j + k]);
				if (space_power > 0)
				{
					for (unsigned int p = 1; p < space_power; ++p)
						aux *= aux;
					S += aux;
				}
				else
				{
					S = std::max(S, aux);
				}
			}

			distance_matrix[num_vectors * j + i] = S;
			distance_matrix[num_vectors * i + j] = S;
		}
}

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
	//Check if supplied data has required format
	if (nlhs > 1)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidOutputArgument", "Multiple output arguments are not supported");

	if  ((nrhs != 2) && (nrhs != 3))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Wrong number of input arguments");

	if (!mxIsFPData(prhs[0]))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "First input argument must have floating point format (either single or double)");

	if (!mxIsFPData(prhs[1]))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Second input argument must have floating point format (either single or double)");

	if ((nrhs == 3) && (!mxIsFPData(prhs[2])))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Third input argument must have floating point format (either single or double)");

	if (mxGetNumberOfDimensions(prhs[0]) > 2)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "First input argument must be a matrix");

	if (mxGetNumberOfDimensions(prhs[1]) > 2)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Second input argument must be a matrix");

	if ((nrhs == 3) && (mxGetNumberOfDimensions(prhs[2]) > 2))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Third input argument must be a matrix");

	const mwSize* input_matrix_dimensions = nullptr;
	input_matrix_dimensions = mxGetDimensions(prhs[0]);
	size_t vector_length = input_matrix_dimensions[0];

	if (nrhs == 3)
	{
		input_matrix_dimensions = mxGetDimensions(prhs[1]);
		if (input_matrix_dimensions[0] != vector_length)
			mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Input arguments must have the same number of rows");
	}

	const mwSize* input_space_power = nullptr;
	const size_t input_space_power_index = (nrhs == 3) ? 2 : 1;
	input_space_power = mxGetDimensions(prhs[input_space_power_index]);

	if ((input_space_power[0] != 1) && (input_space_power[1] != 1))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Space power must be a scalar value");

	unsigned int space_power = 0;
	if (mxIsSingle(prhs[input_space_power_index]))
	{
		float* pData = static_cast<float*>(mxGetData(prhs[input_space_power_index]));
		space_power = static_cast<unsigned int>(pData[0]);
	}
	else
	{
		double* pData = static_cast<double*>(mxGetData(prhs[input_space_power_index]));
		space_power = static_cast<unsigned int>(pData[0]);
	}

	if ((nrhs == 3) && ((mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1])) || (mxIsDouble(prhs[0]) && !mxIsDouble(prhs[1]))))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Both input arguments must use same precision (double or single)");


	size_t num_vectors2 = input_matrix_dimensions[1];
	input_matrix_dimensions = mxGetDimensions(prhs[0]);
	size_t num_vectors1 = input_matrix_dimensions[1];
	mwSize distance_matrix_dimensions[2] = { num_vectors1, num_vectors2 };

	if (mxIsSingle(prhs[0]))
	{
		plhs[0] = mxCreateNumericArray(2, distance_matrix_dimensions, mxSINGLE_CLASS, mxREAL);
		float* pData = static_cast<float*>(mxGetData(plhs[0]));
		if (nrhs == 3)
			computeDistanceMatrix(
				static_cast<float*>(mxGetData(prhs[0])),
				num_vectors1,
				static_cast<float*>(mxGetData(prhs[1])),
				num_vectors2,
				space_power,
				vector_length,
				pData
			);
		else
			computeDistanceMatrix(
				static_cast<float*>(mxGetData(prhs[0])),
				num_vectors1,
				space_power,
				vector_length,
				pData
			);
	}
	else
	{
		plhs[0] = mxCreateNumericArray(2, distance_matrix_dimensions, mxDOUBLE_CLASS, mxREAL);
		double* pData = static_cast<double*>(mxGetData(plhs[0]));
		if (nrhs == 3)
			computeDistanceMatrix(
				static_cast<double*>(mxGetData(prhs[0])),
				num_vectors1,
				static_cast<double*>(mxGetData(prhs[1])),
				num_vectors2,
				space_power,
				vector_length,
				pData
			);
		else
			computeDistanceMatrix(
				static_cast<double*>(mxGetData(prhs[0])),
				num_vectors1,
				space_power,
				vector_length,
				pData
			);
	}
}


bool mxIsFPData(const mxArray* pArray)
{
	return mxIsDouble(pArray) || mxIsSingle(pArray);
}
