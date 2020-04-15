// DistanceMatrixMEX.cpp : Defines the exported functions for the DLL application.
//


#include <mex.h>
#include <cmath>


//Checks if supplied array contains numeric data represented in floating point format. Returns 'true' if so, and 'false' otherwise
bool mxIsFPData(const mxArray* pArray);


//Computes distance matrix for the given sets of vectors
template<typename T>
void computeDistanceMatrix(const T* p_vectors1, size_t num_vectors1, const T* p_vectors2, size_t num_vectors2, size_t vector_length, T* distance_matrix)
{
#pragma omp parallel for
	for (long long j = 0; j < static_cast<long long>(num_vectors2); ++j)
		for (long long i = 0; i < static_cast<long long>(num_vectors1); ++i)
		{
			T S = 0;
			for (size_t k = 0; k < vector_length; ++k)
			{
				T aux = (p_vectors1[vector_length*i + k] - p_vectors2[vector_length*j + k]);
				S += aux*aux;
			}
			distance_matrix[num_vectors1*j + i] = sqrt(S);
		}
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	//Check if supplied data has required format
	if (nlhs > 1)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidOutputArgument", "Multiple output arguments are not supported");

	if (nrhs != 2)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Function accepts exactly 2 input arguments");

	if (!mxIsFPData(prhs[0]))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "First input argument must have floating point format (either single or double)");

	if (!mxIsFPData(prhs[1]))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Second input argument must have floating point format (either single or double)");

	if (mxGetNumberOfDimensions(prhs[0]) > 2)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "First input argument must be a matrix");

	if (mxGetNumberOfDimensions(prhs[1]) > 2)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Second input argument must be a matrix");

	const mwSize* input_matrix_dimensions = nullptr;
	input_matrix_dimensions = mxGetDimensions(prhs[0]);
	size_t vector_length = input_matrix_dimensions[0];

	input_matrix_dimensions = mxGetDimensions(prhs[1]);
	if (input_matrix_dimensions[0] != vector_length)
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Input arguments must have the same number of rows");


	if (mxIsSingle(prhs[0]) && !mxIsSingle(prhs[1]) || mxIsDouble(prhs[0]) && !mxIsDouble(prhs[1]))
		mexErrMsgIdAndTxt("DistanceMatrix:InvalidInputArgument", "Both input arguments must use same precision (double or single)");


	size_t num_vectors2 = input_matrix_dimensions[1];
	input_matrix_dimensions = mxGetDimensions(prhs[0]);
	size_t num_vectors1 = input_matrix_dimensions[1];
	mwSize distance_matrix_dimensions[2] = { num_vectors1, num_vectors2 };

	if (mxIsSingle(prhs[0]))
	{
		plhs[0] = mxCreateNumericArray(2, distance_matrix_dimensions, mxSINGLE_CLASS, mxREAL);
		float* pData = static_cast<float*>(mxGetData(plhs[0]));
		computeDistanceMatrix(static_cast<float*>(mxGetData(prhs[0])), num_vectors1, static_cast<float*>(mxGetData(prhs[1])), num_vectors2, vector_length, pData);
	}
	else
	{
		plhs[0] = mxCreateNumericArray(2, distance_matrix_dimensions, mxDOUBLE_CLASS, mxREAL);
		double* pData = static_cast<double*>(mxGetData(plhs[0]));
		computeDistanceMatrix(static_cast<double*>(mxGetData(prhs[0])), num_vectors1, static_cast<double*>(mxGetData(prhs[1])), num_vectors2, vector_length, pData);
	}
}


bool mxIsFPData(const mxArray* pArray)
{
	return mxIsDouble(pArray) || mxIsSingle(pArray);
}
