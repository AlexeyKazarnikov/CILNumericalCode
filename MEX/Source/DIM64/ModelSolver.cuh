#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NX 64
#define NXSqr 4096
#define NGRID  (NX * NX)
#define HinvSqr 3969
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
	value_type* rhs_v = rhs_u + NGRID;

	//filling shared memory array
	// step 1
	rhs_u[threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + threadIdx.x];
	rhs_v[threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + NGRID + threadIdx.x];

	// step 2
	rhs_u[BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + BSIZE * BSIZE + threadIdx.x];
	rhs_v[BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + NGRID + BSIZE * BSIZE + threadIdx.x];

	// step 3
	rhs_u[2 * BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + 2 * BSIZE * BSIZE + threadIdx.x];
	rhs_v[2 * BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + NGRID + 2 * BSIZE * BSIZE + threadIdx.x];

	// step 4
	rhs_u[3 * BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + 3 * BSIZE * BSIZE + threadIdx.x];
	rhs_v[3 * BSIZE * BSIZE + threadIdx.x] = data[2 * 4 * blockDim.x * blockIdx.x + NGRID + 3 * BSIZE * BSIZE + threadIdx.x];


	unsigned int k[4] = { threadIdx.x, BSIZE * BSIZE + threadIdx.x, 2 * BSIZE * BSIZE + threadIdx.x, 3 * BSIZE * BSIZE + threadIdx.x };
	unsigned int i[4] = { k[0] / NX, k[1] / NX, k[2] / NX, k[3] / NX };
	unsigned int j[4] = { k[0] % NX, k[1] % NX, k[2] % NX, k[3] % NX };

	unsigned int i_prev[4] = {
		(i[0] <= 0)*k[0] + (i[0] > 0)*(k[0] - NX),
		(i[1] <= 0)*k[1] + (i[1] > 0)*(k[1] - NX) ,
		(i[2] <= 0)*k[2] + (i[2] > 0)*(k[2] - NX) ,
		(i[3] <= 0)*k[3] + (i[3] > 0)*(k[3] - NX)
	};

	unsigned int i_next[4] = {
		(i[0] >= NX - 1)*k[0] + (i[0] < NX - 1)*(k[0] + NX),
		(i[1] >= NX - 1)*k[1] + (i[1] < NX - 1)*(k[1] + NX),
		(i[2] >= NX - 1)*k[2] + (i[2] < NX - 1)*(k[2] + NX),
		(i[3] >= NX - 1)*k[3] + (i[3] < NX - 1)*(k[3] + NX)
	};

	unsigned int j_prev[4] = {
		(j[0] < 1)*k[0] + (j[0] >= 1)*(k[0] - 1),
		(j[1] < 1)*k[1] + (j[1] >= 1)*(k[1] - 1),
		(j[2] < 1)*k[2] + (j[2] >= 1)*(k[2] - 1),
		(j[3] < 1)*k[3] + (j[3] >= 1)*(k[3] - 1)
	};

	unsigned int j_next[4] = {
		(j[0] + 1 >= NX)*k[0] + (j[0] + 1 < NX)*(k[0] + 1),
		(j[1] + 1 >= NX)*k[1] + (j[1] + 1 < NX)*(k[1] + 1),
		(j[2] + 1 >= NX)*k[2] + (j[2] + 1 < NX)*(k[2] + 1),
		(j[3] + 1 >= NX)*k[3] + (j[3] + 1 < NX)*(k[3] + 1)
	};

	value_type t_cur = T0;

	__syncthreads();

	while (t_cur < T1)
	{
		value_type K1[8];

		K1[0] = (par.nu1 * HinvSqr)*(rhs_u[j_next[0]] + rhs_u[j_prev[0]] + rhs_u[i_next[0]] + rhs_u[i_prev[0]] - 4 * rhs_u[k[0]])
			+ f(rhs_u[k[0]], rhs_v[k[0]], par);
		K1[1] = (par.nu1 * HinvSqr)*(rhs_u[j_next[1]] + rhs_u[j_prev[1]] + rhs_u[i_next[1]] + rhs_u[i_prev[1]] - 4 * rhs_u[k[1]])
			+ f(rhs_u[k[1]], rhs_v[k[1]], par);
		K1[2] = (par.nu1 * HinvSqr)*(rhs_u[j_next[2]] + rhs_u[j_prev[2]] + rhs_u[i_next[2]] + rhs_u[i_prev[2]] - 4 * rhs_u[k[2]])
			+ f(rhs_u[k[2]], rhs_v[k[2]], par);
		K1[3] = (par.nu1 * HinvSqr)*(rhs_u[j_next[3]] + rhs_u[j_prev[3]] + rhs_u[i_next[3]] + rhs_u[i_prev[3]] - 4 * rhs_u[k[3]])
			+ f(rhs_u[k[3]], rhs_v[k[3]], par);

		K1[4] = (par.nu2 * HinvSqr)*(rhs_v[j_next[0]] + rhs_v[j_prev[0]] + rhs_v[i_next[0]] + rhs_v[i_prev[0]] - 4 * rhs_v[k[0]])
			+ g(rhs_u[k[0]], rhs_v[k[0]], par);
		K1[5] = (par.nu2 * HinvSqr)*(rhs_v[j_next[1]] + rhs_v[j_prev[1]] + rhs_v[i_next[1]] + rhs_v[i_prev[1]] - 4 * rhs_v[k[1]])
			+ g(rhs_u[k[1]], rhs_v[k[1]], par);
		K1[6] = (par.nu2 * HinvSqr)*(rhs_v[j_next[2]] + rhs_v[j_prev[2]] + rhs_v[i_next[2]] + rhs_v[i_prev[2]] - 4 * rhs_v[k[2]])
			+ g(rhs_u[k[2]], rhs_v[k[2]], par);
		K1[7] = (par.nu2 * HinvSqr)*(rhs_v[j_next[3]] + rhs_v[j_prev[3]] + rhs_v[i_next[3]] + rhs_v[i_prev[3]] - 4 * rhs_v[k[3]])
			+ g(rhs_u[k[3]], rhs_v[k[3]], par);
		__syncthreads();

		rhs_u[k[0]] = rhs_u[k[0]] + dT * K1[0];
		rhs_u[k[1]] = rhs_u[k[1]] + dT * K1[1];
		rhs_u[k[2]] = rhs_u[k[2]] + dT * K1[2];
		rhs_u[k[3]] = rhs_u[k[3]] + dT * K1[3];

		rhs_v[k[0]] = rhs_v[k[0]] + dT * K1[4];
		rhs_v[k[1]] = rhs_v[k[1]] + dT * K1[5];
		rhs_v[k[2]] = rhs_v[k[2]] + dT * K1[6];
		rhs_v[k[3]] = rhs_v[k[3]] + dT * K1[7];
		__syncthreads();

		t_cur += dT;
	}



	//finally putting the result into device memory
	// step 1
	data[2 * 4 * blockDim.x * blockIdx.x + threadIdx.x] = rhs_u[threadIdx.x];
	data[2 * 4 * blockDim.x * blockIdx.x + NGRID + threadIdx.x] = rhs_v[threadIdx.x];
	// step 2
	data[2 * 4 * blockDim.x * blockIdx.x + BSIZE * BSIZE + threadIdx.x] = rhs_u[threadIdx.x + BSIZE * BSIZE];
	data[2 * 4 * blockDim.x * blockIdx.x + NGRID + BSIZE * BSIZE + threadIdx.x] = rhs_v[threadIdx.x + BSIZE * BSIZE];
	// step 3
	data[2 * 4 * blockDim.x * blockIdx.x + 2 * BSIZE * BSIZE + threadIdx.x] = rhs_u[threadIdx.x + 2 * BSIZE * BSIZE];
	data[2 * 4 * blockDim.x * blockIdx.x + NGRID + 2 * BSIZE * BSIZE + threadIdx.x] = rhs_v[threadIdx.x + 2 * BSIZE * BSIZE];
	// step 4
	data[2 * 4 * blockDim.x * blockIdx.x + 3 * BSIZE * BSIZE + threadIdx.x] = rhs_u[threadIdx.x + 3 * BSIZE * BSIZE];
	data[2 * 4 * blockDim.x * blockIdx.x + NGRID + 3 * BSIZE * BSIZE + threadIdx.x] = rhs_v[threadIdx.x + 3 * BSIZE * BSIZE];
}

