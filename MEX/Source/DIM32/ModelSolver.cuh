#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NX 32
#define NXSqr 1024
#define NGRID  (NX * NX)
#define HinvSqr 961
#define BSIZE 32

template<typename value_type> struct DeviceModelParameters;

template<typename value_type> __device__ inline value_type f(value_type u, value_type v, DeviceModelParameters<value_type> par);
template<typename value_type> __device__ inline value_type g(value_type u, value_type v, DeviceModelParameters<value_type> par);

template <typename value_type> __device__ inline void ModelSolver(
	value_type* data,
	value_type* sm,
	const DeviceModelParameters<value_type>& par,
	value_type T0,
	value_type T1,
	value_type dT)
{
	//defining shared memory array for r.h.s. elements
	value_type* rhs_u = sm;
	value_type* rhs_v = sm + NGRID;

	unsigned int bid = blockIdx.x;
	unsigned int tid = threadIdx.x;

	//filling shared memory array
	rhs_u[tid] = data[2 * NXSqr * bid + tid];
	rhs_v[tid] = data[2 * NXSqr * bid + NGRID + tid];

	//getting current indexes
	int i = tid / NX;
	int j = tid % NX;

	int i_prev = (i < 1)*tid + (i >= 1)*(tid - NX);
	int i_next = (i + 1 >= NX)*tid + (i + 1 < NX)*(tid + NX);

	int j_prev = (j < 1)*tid + (j >= 1)*(tid - 1);
	int j_next = (j + 1 >= NX)*tid + (j + 1 < NX)*(tid + 1);

	value_type t_cur = T0;

	__syncthreads();

	value_type nu1_mul = par.nu1 * HinvSqr;
	value_type nu2_mul = par.nu2 * HinvSqr;

	while (t_cur < T1)
	{
		rhs_u[tid] = fmaf(
			dT,
			fmaf(
				nu1_mul,
				(rhs_u[j_next] + rhs_u[j_prev] + rhs_u[i_next] + rhs_u[i_prev] - 4 * rhs_u[tid]),
				f(rhs_u[tid], rhs_v[tid], par)),
			rhs_u[tid]);
		rhs_v[tid] = fmaf(
			dT,
			fmaf(
				nu2_mul,
				(rhs_v[j_next] + rhs_v[j_prev] + rhs_v[i_next] + rhs_v[i_prev] - 4 * rhs_v[tid]),
				g(rhs_u[tid], rhs_v[tid], par)),
			rhs_v[tid]);
		__syncthreads();

		t_cur += dT;
	}

	//finally putting the result into device memory
	data[2 * NXSqr * bid + tid] = rhs_u[tid];
	data[2 * NXSqr * bid + NGRID + tid] = rhs_v[tid];
}
