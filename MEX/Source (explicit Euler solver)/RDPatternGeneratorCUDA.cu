#include "RDSystem.cuh"
#include "ModelSolver.cuh"
#include "RunModelSolver.cuh"

#include "mex.h"

#include <vector>

using namespace std;


void mexFunction(int nlhs, mxArray *plhs[],
	int nrhs, const mxArray *prhs[])
{
	//First, check input arguments count
	if (nrhs < 6)
		mexErrMsgTxt("Wrong number of input arguments. ");

	//Input data should have the following format
	//1: T0:double - starting point of time interval
	//2: T1:double - ending point of time interval
	//3: dT:double - "safe" time interval
	//4: time_step:double - time step value
	//5: SystemParameters:struct - parameters of the system
	//6: InitialConditionVector: single array - vector of initial conditions
	//7: PatternCount:double - required number of patterns
	//8: devices: double array - CUDA devices to use

	//First, we will check input data

	if (nrhs != 8)
	{
		mexErrMsgTxt("The function must have exactly 8 input arguments!");
	}

	//Input parameters 1-4
	for (int i = 0; i < 4; ++i)
	{
		if (!(mxIsDouble(prhs[i])) || (mxGetNumberOfDimensions(prhs[i]) != 2) || (mxGetNumberOfElements(prhs[i]) != 1))
		{
			mexErrMsgTxt("First four input parameters must be scalar double values!");
		}
	}

	//Input parameters 5
	if (!mxIsStruct(prhs[4]))
	{
		mexErrMsgTxt("5-th input parameters must be struct!");
	}

	//Input parameter 6
	if (!(mxIsSingle(prhs[5])))
	{
		mexErrMsgTxt("6-th input parameters must be single array!");
	}

	//Input parameter 7
	if (!(mxIsDouble(prhs[6])) || (mxGetNumberOfDimensions(prhs[6]) != 2) || (mxGetNumberOfElements(prhs[6]) != 1))
	{
		mexErrMsgTxt("7-th input parameter must be scalar double value!");
	}

	//Input parameter 8
	if (mxIsEmpty(prhs[7]))
		mexErrMsgTxt("8-th input parameter cannot be empty!");


	//Input parameters 1-4
	float T0 = static_cast<float>(mxGetScalar(prhs[0]));
	float T1 = static_cast<float>(mxGetScalar(prhs[1]));
	float dT = static_cast<float>(mxGetScalar(prhs[2]));
	float time_step = static_cast<float>(mxGetScalar(prhs[3]));

	//Input parameter 5 (system parameters)
	DeviceModelParameters<float> par;

	vector<float> parValues(par.ParameterCount);
	for (int k = 0; k < par.ParameterCount; ++k)
	{
		mxArray *tmp = mxGetField(prhs[4], 0, par.ParameterNames[k]);
		if (tmp == NULL)
			mexErrMsgTxt("Field names in parameter 5 are not correct!");
		parValues[k] = static_cast<float>(mxGetScalar(tmp));
	}
	par.Initialize(parValues.data());

	//Input parameter 6 (initial conditions)
	float* InitialConditionVector = mxGetSingles(prhs[5]);

	//Input parameter 7 (pattern count)
	size_t PatternCount = static_cast<unsigned int>(mxGetScalar(prhs[6]));

	//Input parameter 8 (CUDA devices)
	size_t DeviceCount = mxGetNumberOfElements(prhs[7]);
	double* device_data = mxGetDoubles(prhs[7]);

	if (DeviceCount > PatternCount)
		DeviceCount = PatternCount;

	vector<int> devices(DeviceCount);
	for (int k = 0; k < DeviceCount; ++k)
		devices[k] = static_cast<int>(device_data[k]);


	//Next we will check output data
	if (nlhs != 1)
		mexErrMsgTxt("The function must have exactly one output argument!");

	float* data;
	mwSize dims[2] = { 2 * NX * NX, PatternCount };
	plhs[0] = mxCreateNumericArray(2, dims, mxSINGLE_CLASS, mxREAL);
	data = mxGetSingles(plhs[0]);

	LaunchDataContainer launchdata;
	launchdata.BlockNumber = PatternCount;
	launchdata.SharedMemoryPerBlock = 2 * NX * NX;
	launchdata.ThreadsPerBlock = BSIZE * BSIZE;

	SolverParameters<float> solpar;
	solpar.T0 = T0;
	solpar.T1 = T1;
	solpar.dt = time_step;

	auto result = RunModelSolver(
		launchdata, // data, which determines how CUDA kernel call should be configured
		InitialConditionVector, // pointer to the array, where initial data for simulations is stored
		2 * NX * NX * PatternCount, // size (in elements) of the initialdata array
		data, // pointer to the array, where final data should be stored. Assumed to have the same size as initialdata
		par,
		solpar,
		devices,
		false);

	if (result.message != "")
		mexPrintf(result.message.data());

	return;
}