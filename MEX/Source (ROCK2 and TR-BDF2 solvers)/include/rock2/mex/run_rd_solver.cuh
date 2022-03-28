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

#include "utils/MemoryLink.h"
#include "utils/MexUtils.h"
#include "utils/TestUtils.h"

template <typename value_type, typename index_type, typename gpu_handler> void run_rd_solver(
	int nlhs, 
	mxArray* plhs[],
	int nrhs, 
	const mxArray* prhs[],
	const std::vector<value_type>& par_array,
	index_type sys_dim,
	value_type nu1,
	value_type nu2
)
{
	std::vector<std::string> sol_par_fields = {
		"initial_time_step", 
		"final_time_point",
		"conv_norm",
		"conv_stage_number",
		"abs_tol", 
		"rel_tol",
		"grid_resolution"
	};
	std::vector<double> sol_par_values(sol_par_fields.size());
	checkmxStruct(prhs[1], sol_par_fields, sol_par_values, "Input parameter sol_par must be a struct with correct fields!");
	
	value_type initial_time_step = sol_par_values[0];
	value_type final_time_point = sol_par_values[1];
	value_type conv_norm = sol_par_values[2];
	index_type conv_stage_number = sol_par_values[3];
	value_type abs_tol = sol_par_values[4];
	value_type rel_tol = sol_par_values[5];
	value_type grid_resolution_value = sol_par_values[6];
	
	index_type grid_resolution = static_cast<index_type>(grid_resolution_value);
	
	index_type sys_size = 2;
	for (auto k = 0; k < sys_dim; ++k)
		sys_size *= grid_resolution;

	double Npatterns;
	checkmxScalar(prhs[2], Npatterns, "Input parameter Npatterns must be a scalar value!");
	
	index_type sim_number = static_cast<index_type>(Npatterns);

	double* IC;
	checkmxArray(prhs[3], IC, sys_size * sim_number, "Input parameter IC must be a numeric array with correct number of elements!");

	double* devices_ptr;
	checkmxArray(prhs[4], devices_ptr, "Input parameter devices must be a numeric (double) array!");

	unsigned int devices_length = static_cast<unsigned int>(mxGetNumberOfElements(prhs[4]));
	std::vector<unsigned int> devices;
	for (unsigned int k = 0; k < devices_length; ++k)
	{
		devices.push_back(static_cast<unsigned int>(devices_ptr[k]));
	}
	if (devices.size() > sim_number)
		devices.resize(sim_number);

	if (nlhs != 1)
		mexErrMsgTxt("Wrong number of output arguments (must be 1!)");

	double* result;
	mwSize dims[2] = {sys_size, sim_number };

	plhs[0] = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
	result = mxGetDoubles(plhs[0]);

	// executing solver in multiple threads
	std::vector<cudaError_t> cuda_results(devices.size());
	std::vector<bool> warning_results(devices.size());
	std::vector<std::shared_ptr<std::thread>> threads(devices.size());

	for (unsigned int k = 0; k < devices.size(); ++k)
	{
		threads[k] = std::make_shared<std::thread>(
			[k, sim_number, grid_resolution, nu1, nu2, par_array, initial_time_step, final_time_point, conv_norm, conv_stage_number, abs_tol, rel_tol, IC, result, sys_size,
			&cuda_results, &warning_results, &devices]()
			{
				auto curr_device_number = devices[k];

				index_type start_sim_index = k * sim_number / devices.size();
				index_type end_sim_index = (k + 1) * sim_number / devices.size();
				index_type curr_sim_number = end_sim_index - start_sim_index;

				index_type sim_size = sys_size;

				MemoryLink<value_type> curr_input_data(IC + start_sim_index * sim_size, curr_sim_number * sim_size);
				MemoryLink<value_type> curr_output_data(result + start_sim_index * sim_size, curr_sim_number * sim_size);

				std::vector<std::shared_ptr<R2Runner<value_type, index_type>>> runners(curr_sim_number);
				for (unsigned int j = 0; j < runners.size(); ++j)
					runners[j] = std::make_shared<R2Runner<value_type, index_type>>(
						j,
						initial_time_step,
						final_time_point,
						conv_norm,
						conv_stage_number);

				R2GPUScheduler<value_type, index_type> scheduler(curr_sim_number, curr_input_data, curr_output_data, curr_device_number);
				gpu_handler handler(curr_sim_number, sim_size, grid_resolution, nu1, nu2, par_array, curr_device_number, abs_tol, rel_tol);

				while (std::any_of(runners.begin(), runners.end(), [](std::shared_ptr<R2Runner<value_type, index_type>>& val) {return val->is_active(); }))
				{
					for (auto& runner : runners)
						runner->run(scheduler);

					scheduler.send_data_to_device();

					if ((cuda_results[k] = handler.handle_computations(scheduler)) != cudaSuccess)
					{
						break;
					}

					scheduler.receive_data_from_device();
					scheduler.clear_queues();
				}
				scheduler.finalize();
				
				if (std::any_of(runners.begin(), runners.end(), [](std::shared_ptr<R2Runner<value_type, index_type>>& val) {return !val->is_convergence_reached();}))
					warning_results[k] = true;
			});		
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
		if (cuda_results[k] != cudaSuccess)
		{
			std::ostringstream os;
			os << "Thread " << k << ": CUDA error " << cuda_results[k] << std::endl;
			mexWarnMsgTxt(os.str().data());
			errorFlag = true;
		}
		
		if (warning_results[k])
		{
			std::ostringstream os;
			os << "Thread " << k << ": Some of the simulations did not converge! Please check solver settings!" << std::endl;
			mexWarnMsgTxt(os.str().data());
		}
	}

	if (errorFlag)
		mexErrMsgTxt("An error occured while executing the MEX-file!");
	
}