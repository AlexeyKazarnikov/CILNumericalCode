#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

template<typename value_type> struct DeviceModelParameters;

template<typename value_type> inline value_type f(value_type u, value_type v, DeviceModelParameters<value_type> par);

template<typename value_type> inline value_type g(value_type u, value_type v, DeviceModelParameters<value_type> par);


template <typename value_type> void RunEulerIntegration(
	const value_type T0, 
	const value_type T1, 
	const value_type dt, 
	value_type* w,
	value_type* dw,
	DeviceModelParameters<value_type> par,
	const size_t N)
{
	auto HinvSqr = (N - 1) * (N - 1);
	auto NGRID = N * N;

	value_type t = T0;

	while (t < T1)
	{
		for (unsigned int k = 0; k < NGRID; ++k)
		{
			unsigned int i = k / N;
			unsigned int j = k % N;

			unsigned int i_prev = (i < 1)*k + (i >= 1)*(k - N);
			unsigned int i_next = (i + 1 >= N)*k + (i + 1 < N)*(k + N);

			unsigned int j_prev = (j < 1)*k + (j >= 1)*(k - 1);
			unsigned int j_next = (j + 1 >= N)*k + (j + 1 < N)*(k + 1);

			dw[k] = par.nu1 * HinvSqr * (w[i_next] + w[i_prev] + w[j_next] + w[j_prev] - 4 * w[k]) + f(w[k], w[k + NGRID], par);
			dw[k + NGRID] = par.nu2 * HinvSqr * (w[i_next + NGRID] + w[i_prev + NGRID] + w[j_next + NGRID] + w[j_prev + NGRID] - 4 * w[k + NGRID]) + g(w[k], w[k + NGRID], par);
		}
		
		for (unsigned int k = 0; k < NGRID; ++k)
		{
			w[k] += dt * dw[k];
			w[k + NGRID] += dt * dw[k + NGRID];
		}

		t += dt;
	}

}


template <typename value_type> void RunModelSolver(
	const value_type T0,
	const value_type T1,
	const value_type dt,
	const value_type* ic,
	value_type* result,
	DeviceModelParameters<value_type> par,
	const size_t N,
	const size_t Nsim)
{
	vector<value_type> lm(2 * N * N, 0);

	auto NDIM = 2 * N * N;

	for (unsigned int k = 0; k < Nsim; ++k)
	{
		std::memcpy(&result[NDIM * k], &ic[NDIM * k], NDIM * sizeof(value_type));
		RunEulerIntegration(T0, T1, dt, &result[NDIM * k], lm.data(), par, N);
	}
}



