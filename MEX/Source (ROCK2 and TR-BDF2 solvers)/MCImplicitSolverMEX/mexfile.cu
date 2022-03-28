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

#include "mc_models/mc/mc_model_generic.cuh"
#include "utils/fds_utils_neumann.h"
#include "mc_models/run_model_solver.cuh"
#include "utils/MexUtils.h"

using namespace std;

void mexFunction(int nlhs, mxArray* plhs[],
	int nrhs, const mxArray* prhs[])
{
	vector<string> par_fields = { "alpha", "beta", "delta", "L", "D", "tau", "N" };
	vector<double> par_values(par_fields.size());
	checkmxStruct(prhs[0], par_fields, par_values, "Input parameter par must be a struct with correct fields!");

	ModelParameters<double> par(
		par_values[0],
		par_values[1],
		par_values[2],
		par_values[3],
		par_values[4],
		par_values[5],
		static_cast<unsigned int>(par_values[6]));

	//mexPrintf("alpha = %f, beta = %f, delta = %f, L = %f, D = %f, tau = %f, N = %i \n",
	//	par.alpha, par.beta, par.delta, par.L, par.D, par.tau, par.N);

	vector<string> sol_par_fields = {
		"dt", 
		"omega", 
		"abs_tol", 
		"max_iter", 
		"conv_norm", 
		"jac_step", 
		"max_time", 
		"max_step", 
		"max_step_number", 
		"rel_tol",
		"jac_update_interval"
	};
	vector<double> sol_par_values(sol_par_fields.size());
	checkmxStruct(prhs[1], sol_par_fields, sol_par_values, "Input parameter sol_par must be a struct with correct fields!");

	SolverParameters<double> sol_par;
	sol_par.initial_step = sol_par_values[0];
	sol_par.omega = sol_par_values[1];
	sol_par.abs_tol = sol_par_values[2];
	sol_par.max_iter = static_cast<unsigned int>(sol_par_values[3]);
	sol_par.conv_norm = sol_par_values[4];
	sol_par.jac_step = sol_par_values[5];
	sol_par.max_time = sol_par_values[6];
	sol_par.max_step = sol_par_values[7];
	sol_par.max_step_number = static_cast<unsigned int>(sol_par_values[8]);
	sol_par.rel_tol = sol_par_values[9];
	sol_par.jac_update_interval = static_cast<unsigned int>(sol_par_values[10]);

	//mexPrintf(
	//	"dt = %f, omega = %f, abs_tol = %f, max_iter = %i, conv_norm = %f, jac_step = %f, max_time = %f, max_step = %f, max_step_number = %i \n",
	//	sol_par.initial_step, sol_par.omega, sol_par.abs_tol, sol_par.max_iter, sol_par.conv_norm, sol_par.jac_step, sol_par.max_time, sol_par.max_step, sol_par.max_step_number);

	double Npatterns;
	checkmxScalar(prhs[2], Npatterns, "Input parameter Npatterns must be a scalar value!");

	//mexPrintf("Nsim = %i \n", Npatterns);

	double* IC;
	checkmxArray(prhs[3], IC, (2 * par.N + 1) * static_cast<unsigned int>(Npatterns), "Input parameter IC must be a numeric array with correct number of elements!");

	double* devices_ptr;
	checkmxArray(prhs[4], devices_ptr, "Input parameter devices must be a numeric (double) array!");

	unsigned int devices_length = static_cast<unsigned int>(mxGetNumberOfElements(prhs[4]));
	vector<unsigned int> devices;
	for (unsigned int k = 0; k < devices_length; ++k)
	{
		devices.push_back(static_cast<unsigned int>(devices_ptr[k]));
	}

	if (nlhs != 1)
		mexErrMsgTxt("Wrong number of output arguments (must be 1!)");

	double* result;
	mwSize dims[2] = { (2 * par.N + 1), static_cast<unsigned int>(Npatterns) };


	plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	result = mxGetDoubles(plhs[0]);

	// executing solver in multiple threads

	std::vector<RunResult> results(devices.size());
	std::vector<std::shared_ptr<std::thread>> threads(devices.size());

	for (unsigned int k = 0; k < devices.size(); ++k)
	{
		double* w_start = IC + k * static_cast<unsigned int>(Npatterns) / devices.size() * (2 * par.N + 1);
		double* output_start = result + k * static_cast<unsigned int>(Npatterns) / devices.size() * (2 * par.N + 1);
		unsigned int num_sim = k < static_cast<unsigned int>(devices.size()) - 1 ?
			static_cast<unsigned int>(Npatterns) / devices.size()
			: static_cast<unsigned int>(Npatterns) - static_cast<unsigned int>(Npatterns) / devices.size() * (devices.size() - 1);

		threads[k] = std::make_shared<std::thread>(
			run_model_solver_thread<double>,
			w_start, 
			output_start, 
			par.sys_size,
			num_sim, 
			devices[k], 
			par, 
			sol_par, 
			std::ref(results[k])
			);
	}

	for (unsigned int k = 0; k < devices.size(); ++k)
	{
		threads[k]->join();
	}
	
	for (unsigned int k = 0; k < devices.size(); ++k)
	{
		CUDA_LOG(cudaSetDevice(devices[k]));
		CUDA_LOG(cudaDeviceReset());
	}

	bool errorFlag = false;

	for (unsigned int k = 0; k < devices.size(); ++k)
	{
		if (results[k].code != cudaSuccess)
		{
			std::ostringstream os;
			os << "Thread " << k << ": CUDA error " << results[k].code << ", message: " << results[k].message << std::endl;
			mexWarnMsgTxt(os.str().data());
			errorFlag = true;
		}

		if ((results[k].code == cudaSuccess) && (!results[k].message.empty()))
		{
			std::ostringstream os;
			os << "Thread " << k << ": warning: " << results[k].message << std::endl;
			mexWarnMsgTxt(os.str().data());
		}
	}

	if (errorFlag)
		mexErrMsgTxt("An error occured while executing the MEX-file (see the details above)!");
}