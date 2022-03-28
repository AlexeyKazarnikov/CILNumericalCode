#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mex.h"

#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "MexUtils.h"
#include "RunDeviceComputations.cuh"


using namespace std;

void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	if (nrhs != 2)
		mexErrMsgTxt("Function must have exactly two input arguments!");
	if (nlhs != 1)
		mexErrMsgTxt("Function must have exactly one output argument!");

	if ( !( (mxIsSingle(prhs[0]) && mxIsSingle(prhs[1])) || (mxIsDouble(prhs[0]) && mxIsDouble(prhs[1])) ) )
	{
		mexErrMsgTxt("Both input arguments must be single or double arrays!");
	}

	const mwSize* first_matrix_size = mxGetDimensions(prhs[0]);
	const mwSize* second_matrix_size = mxGetDimensions(prhs[1]);

	if (first_matrix_size[0] != second_matrix_size[0])
		mexErrMsgTxt("The first dimension of both input arrays must be the same!");

	const size_t dimension = first_matrix_size[0];

	RunResult result;

	if (mxIsSingle(prhs[0]))
	{
		float* first_matrix_data = nullptr;
		float* second_matrix_data = nullptr;

		checkmxArray(prhs[0], first_matrix_data, "input parameter 1 must be a matrix!");
		checkmxArray(prhs[1], second_matrix_data, "input parameter 2 must be a matrix!");

		mwSize output_dims[2] = { first_matrix_size[1], second_matrix_size[1] };
		plhs[0] = mxCreateNumericArray(2, output_dims, mxSINGLE_CLASS, mxREAL);
		float* output_data = (float*)mxGetSingles(plhs[0]);
		
		MemoryLink<float> first_matrix_link {first_matrix_data, first_matrix_size[0] * first_matrix_size[1]};
		MemoryLink<float> second_matrix_link {second_matrix_data, second_matrix_size[0] * second_matrix_size[1]};
		MemoryLink<float> output_link {output_data, first_matrix_size[1] * second_matrix_size[1]};

		result = runDeviceComputations<float>(
			first_matrix_link,
			second_matrix_link,
			output_link,
			dimension
		);
	}
	else
	{
		double* first_matrix_data = nullptr;
		double* second_matrix_data = nullptr;

		checkmxArray(prhs[0], first_matrix_data, "input parameter 1 must be a matrix!");
		checkmxArray(prhs[1], second_matrix_data, "input parameter 2 must be a matrix!");

		mwSize output_dims[2] = { first_matrix_size[1], second_matrix_size[1] };
		plhs[0] = mxCreateNumericArray(2, output_dims, mxDOUBLE_CLASS, mxREAL);
		double* output_data = (double*)mxGetDoubles(plhs[0]);
		
		MemoryLink<double> first_matrix_link {first_matrix_data, first_matrix_size[0] * first_matrix_size[1]};
		MemoryLink<double> second_matrix_link {second_matrix_data, second_matrix_size[0] * second_matrix_size[1]};
		MemoryLink<double> output_link {output_data, first_matrix_size[1] * second_matrix_size[1]};

		result = runDeviceComputations<double>(
			first_matrix_link,
			second_matrix_link,
			output_link,
			dimension
		);
	}
	
	cudaDeviceReset();

	if (result.code != 0)
	{
		ostringstream os;
		os << "Device computations error: " << result;
		mexErrMsgTxt(os.str().data());
	}
}