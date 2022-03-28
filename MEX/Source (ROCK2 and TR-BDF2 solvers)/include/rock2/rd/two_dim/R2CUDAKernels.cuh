# pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "rock2/rd/rd_rhs_generic.h"
#include "utils/reduce_utils.cuh"

# define constant_type double

__constant__ constant_type dc_nu1_hinv2;

__constant__ constant_type dc_nu2_hinv2;


template <typename value_type, typename index_type>
__device__ __inline__ void eval_model_rhs(
    const value_type* y,
    value_type& rhs_u,
    value_type& rhs_v,
    value_type& u, 
    value_type& v,
    const index_type gid,
    const index_type gid_i,
    const index_type gid_j,
    const index_type grid_resolution,
    const index_type grid_size
)
{
    auto iprev = (gid_j == 0) * gid + (gid_j > 0) * (gid - grid_resolution);
    auto inext = (gid_j < grid_resolution - 1) * (gid + grid_resolution) + (gid_j == grid_resolution - 1) * gid;
    auto jprev = (gid_i == 0) * gid + (gid_i > 0) * (gid - 1);
    auto jnext = (gid_i == grid_resolution - 1) * gid + (gid_i < grid_resolution - 1) * (gid + 1);

    u = y[gid];
    v = y[gid + grid_size];

    rhs_u = dc_nu1_hinv2 * (y[iprev] + y[inext] + y[jprev] + y[jnext] - 4 * u)
        + f(u, v, dc_par);

    rhs_v = dc_nu2_hinv2 * (y[iprev + grid_size] + y[inext + grid_size] + y[jprev + grid_size] + y[jnext + grid_size] - 4 * v)
        + g(u, v, dc_par);
}



// yjm2 = yprev; yjm1 = yprev + dt * recf(mr) * f(y_prev); if mdeg < 2 y = yjm1;
template <typename value_type, typename index_type>
__global__ void r2_initial_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed
    auto gid = fid % grid_size; // index of the node in spatial grid, which is being processed
    auto sys_size = 2 * grid_size;

    if (mid < sim_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        auto sid = t_run_indices[3 * mid];
        auto pos_recf = t_run_indices[3 * mid + 1];
        auto degree = t_run_indices[3 * mid + 2];

        value_type* y = t_sys_state + 4 * sys_size * sid;
        value_type* y_prev = y + sys_size;
        value_type* yjm1 = y_prev + sys_size;
        value_type* yjm2 = yjm1 + sys_size;
        value_type time_step = t_time_steps[sid];

        // yjm2 = y_prev
        yjm2[gid] = y_prev[gid];
        yjm2[gid + grid_size] = y_prev[gid + grid_size];

        // yjm1 = yprev + dt * recf(mr) * f(y_prev)
        //eval_model_rhs(y_prev, yjm1, gid, grid_resolution, grid_size);
        //{
            auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
            auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
            auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
            auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

            value_type u = y_prev[gid];
            value_type v = y_prev[gid + grid_size];

            value_type rhs_u = dc_nu1_hinv2 * (y_prev[iprev] + y_prev[inext] + y_prev[jprev] + y_prev[jnext] - 4 * u)
                + f(u, v, dc_par);

            value_type rhs_v = dc_nu2_hinv2 * 
                (y_prev[iprev + grid_size] + y_prev[inext + grid_size] + y_prev[jprev + grid_size] + y_prev[jnext + grid_size] - 4 * v)
                + g(u, v, dc_par);
        //}

        value_type out_u = rhs_u * time_step * dc_recf[pos_recf] + u;
        value_type out_v = rhs_v * time_step * dc_recf[pos_recf] + v;
        yjm1[gid] = out_u;
        yjm1[gid + grid_size] = out_v;

        // if mdeg < 2 y = yjm1;
        if (degree < 2)
        {
            y[gid] = out_u;
            y[gid + grid_size] = out_v;
        }

    } // if
}


template <typename value_type, typename index_type>
__global__ void r2_rec_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto gid = grid_resolution * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_i = blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_j = blockDim.y * blockIdx.y + threadIdx.y;
    auto sys_size = 2 * grid_size;

    if (gid_i >= grid_resolution || gid_j >= grid_resolution)
        return;

    auto sid = t_run_indices[3 * blockIdx.z];
    auto pos_recf = t_run_indices[3 * blockIdx.z + 1];
    auto counter = t_run_indices[3 * blockIdx.z + 2];

    value_type* y_start = t_sys_state + 4 * sys_size * sid;

    auto y_shift = 2 - (counter) % 3;
    auto yjm1_shift = 2 - (counter - 1) % 3;
    auto yjm2_shift = 2 - (counter - 2) % 3;

    value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * sys_size;
    value_type* yjm1 = y_start + (yjm1_shift > 0) * (yjm1_shift + 1) * sys_size;
    value_type* yjm2 = y_start + (yjm2_shift > 0) * (yjm2_shift + 1) * sys_size;

    value_type time_step = t_time_steps[sid];

    value_type temp1 = time_step * dc_recf[pos_recf + 2 * (counter - 2) + 1];
    value_type temp3 = -dc_recf[pos_recf + 2 * (counter - 2) + 2];
    value_type temp2 = 1 - temp3;

    // y=temp1*f(yjm1)+temp2*yjm1+temp3*yjm2;
    value_type u, v, rhs_u, rhs_v;

    eval_model_rhs(yjm1, rhs_u, rhs_v, u, v, gid, gid_i, gid_j, grid_resolution, grid_size);

    y[gid] = temp1 * rhs_u + temp2 * u + temp3 * yjm2[gid];
    y[gid + grid_size] = temp1 * rhs_v + temp2 * v + temp3 * yjm2[gid + grid_size];
  
}

// finishing procedure
// temp1 = h * fp1(mz);
// temp2 = h * fp2(mz);

// yjm2 = f(neqn, y);
// yjm1 = y + temp1 * yjm2;
// y = f(neqn, yjm1);

// temp3 = temp2 * (y - yjm2);
// y = yjm1 + temp1 * y + temp3;
template <typename value_type, typename index_type>
__global__ void r2_fp1_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto gid = grid_resolution * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_i = blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_j = blockDim.y * blockIdx.y + threadIdx.y;
    auto sys_size = 2 * grid_size;

    if (gid_i >= grid_resolution || gid_j >= grid_resolution)
        return;

    auto sid = t_run_indices[2 * blockIdx.z];
    auto pos_fp = t_run_indices[2 * blockIdx.z + 1];
    auto counter = dc_ms[pos_fp];
    if (counter < 2)
        counter = 2;

    value_type* y_start = t_sys_state + 4 * sys_size * sid;

    auto y_shift = 2 - (counter) % 3;
    auto yjm1_shift = 2 - (counter - 1) % 3;
    auto yjm2_shift = 2 - (counter - 2) % 3;

    value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * sys_size;
    value_type* yjm1 = y_start + (yjm1_shift > 0) * (yjm1_shift + 1) * sys_size;
    value_type* yjm2 = y_start + (yjm2_shift > 0) * (yjm2_shift + 1) * sys_size;

    value_type time_step = t_time_steps[sid];

    value_type temp1 = time_step * dc_fp1[pos_fp];
    value_type temp2 = time_step * dc_fp2[pos_fp];

    // yjm2[reg] = f(neqn, y->[reg]); 
    value_type yjm2_u, yjm2_v, y_u, y_v;
    eval_model_rhs(y, yjm2_u, yjm2_v, y_u, y_v, gid, gid_i, gid_j, grid_resolution, grid_size);

    // yjm2 = yjm2[reg]
    yjm2[gid] = yjm2_u;
    yjm2[gid + grid_size] = yjm2_v;

    // yjm1 = y[reg] + temp1*yjm2[reg]; 
    yjm1[gid] = y_u + temp1 * yjm2_u;
    yjm1[gid + grid_size] = y_v + temp1 * yjm2_v;
}

template <typename value_type, typename index_type>
__global__ void r2_fp2_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto gid = grid_resolution * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_i = blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_j = blockDim.y * blockIdx.y + threadIdx.y;
    auto sys_size = 2 * grid_size;

    if (gid_i >= grid_resolution || gid_j >= grid_resolution)
        return;

    auto sid = t_run_indices[2 * blockIdx.z];
    auto pos_fp = t_run_indices[2 * blockIdx.z + 1];
    auto counter = dc_ms[pos_fp];
    if (counter < 2)
        counter = 2;

    value_type* y_start = t_sys_state + 4 * sys_size * sid;

    auto y_shift = 2 - (counter) % 3;
    auto yjm1_shift = 2 - (counter - 1) % 3;
    auto yjm2_shift = 2 - (counter - 2) % 3;

    value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * sys_size;
    value_type* yjm1 = y_start + (yjm1_shift > 0) * (yjm1_shift + 1) * sys_size;
    value_type* yjm2 = y_start + (yjm2_shift > 0) * (yjm2_shift + 1) * sys_size;

    value_type time_step = t_time_steps[sid];

    value_type temp1 = time_step * dc_fp1[pos_fp];
    value_type temp2 = time_step * dc_fp2[pos_fp];

    // y[reg] = f(neqn,yjm1->[reg]);
    value_type yjm1_u, yjm1_v, y_u, y_v;
    eval_model_rhs(yjm1, y_u, y_v, yjm1_u, yjm1_v, gid, gid_i, gid_j, grid_resolution, grid_size);

    // temp3 = temp2 * (y[reg]-yjm2[reg]);
    // y = yjm1 + temp1 * y[reg] + temp3;
    // yjm2 = yjm2[reg]
    // yjm1 = y[reg]
    value_type temp3_u = temp2 * (y_u - yjm2[gid]);
    value_type temp3_v = temp2 * (y_v - yjm2[gid + grid_size]);

    y[gid] = yjm1_u + temp1 * y_u + temp3_u;
    y[gid + grid_size] = yjm1_v + temp1 * y_v + temp3_v;

    yjm2[gid] = temp3_u;
    yjm2[gid + grid_size] = temp3_v;
}


template <typename value_type, typename index_type>
__global__ void spectral_radius_est_stage(
    const value_type* t_sys_state,
    value_type* t_rho,
    const index_type* t_run_indices,
    const index_type grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    static __shared__ value_type shared_mem[32];

    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto sid = t_run_indices[blockIdx.x];
    auto sys_size = 2 * grid_size;
    value_type val = 0;
 
    const value_type* y = t_sys_state + 4 * sys_size * sid + sys_size;

    auto gid = tid;
    while (gid < grid_size)
    {
        auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
        auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
        auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
        auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

        unsigned int lin_coeff = 4 - (iprev == gid) - (inext == gid) - (jprev == gid) - (jnext == gid);

        auto center_u = -dc_nu1_hinv2 * lin_coeff + dfdu(y[gid], y[gid + grid_size], dc_par);
        auto center_v = -dc_nu2_hinv2 * lin_coeff + dgdv(y[gid], y[gid + grid_size], dc_par);
        auto radius_u = dc_nu1_hinv2 * lin_coeff + abs(dfdv(y[gid], y[gid + grid_size], dc_par));
        auto radius_v = dc_nu2_hinv2 * lin_coeff + abs(dgdu(y[gid], y[gid + grid_size], dc_par));
        auto res_gid = max(abs(center_u) + radius_u, abs(center_v) + radius_v);
        // auto res_gid = max(max(abs(center_u + radius_u), abs(center_u - radius_u)), max(abs(center_v + radius_v), abs(center_v - radius_v)));
        val = max(val, res_gid);

        gid += blockDim.x;
    } // while
    
    __syncthreads();

    // reduction

    // indexes for the current thread
    const unsigned int lane = tid % warpSize;
    const unsigned int wid = tid / warpSize;

    // warp-level reduction
    val = warp_reduce_max(val);

    // writing reduced values to shared memory
    if (lane == 0)
        shared_mem[wid] = val;

    __syncthreads();

    //read from shared memory only if that warp existed
    val = (tid < blockDim.x / warpSize) ? shared_mem[lane] : 0;

    if (wid == 0)
        val = warp_reduce_max(val); //Final reduce within first warp

    if (tid == 0)
    {
        t_rho[blockIdx.x] = val;
    }
}

