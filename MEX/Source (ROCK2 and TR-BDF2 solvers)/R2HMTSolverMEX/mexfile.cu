#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "mex.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "rock2/R2Runner.cuh"
#include "rock2/R2GPUScheduler.cuh"
#include "rock2/rd/one_dim/RDGPUHandler1D.cuh"
#include "rock2/rd/models/rd_rhs_hmt.h"

#include "utils/MemoryLink.h"
#include "utils/MexUtils.h"
#include "utils/TestUtils.h"

#include "rock2/mex/run_rd_solver.cuh"

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[])
{
	using value_type = double;
	using index_type = unsigned short;

	vector<string> par_fields = { "nu1", "nu2", "m1", "m2", "m3", "k" };
	vector<double> par_values(par_fields.size());
	checkmxStruct(prhs[0], par_fields, par_values, "Input parameter par must be a struct with correct fields!");
	
	value_type nu1 = par_values[0];
	value_type nu2 = par_values[1];

	ModelParameters<double> par;
	par.m1 = par_values[2];
	par.m2 = par_values[3];
	par.m3 = par_values[4];
	par.k = par_values[5];
	
	std::vector<value_type> par_array(10);
	par_to_array(par, par_array.data());

	run_rd_solver<value_type,index_type,RDGPUHandler1D<value_type,index_type>>(
		nlhs, 
		plhs,
		nrhs, 
		prhs,
		par_array,
		1,
		nu1,
		nu2
	);
}