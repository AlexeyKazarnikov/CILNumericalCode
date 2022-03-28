# pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "rock2/rd/rd_rhs_generic.h"
#include "utils/reduce_utils.cuh"

__constant__ constant_type dc_nu1_hinv2;

__constant__ constant_type dc_nu2_hinv2;

template <typename value_type, typename index_type>
__device__ __inline__ void eval_model_rhs(
    const value_type* y,
    value_type* rhs,
    const index_type gid,
    const index_type grid_resolution,
    const index_type grid_size
)
{
    auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
    auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
    auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
    auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

    rhs[gid] = dc_nu1_hinv2 * (y[iprev] + y[inext] + y[jprev] + y[jnext] - 4 * y[gid])
        + f(y[gid], y[gid + grid_size], dc_par);

    rhs[gid + grid_size] = dc_nu2_hinv2 * (y[iprev + grid_size] + y[inext + grid_size] + y[jprev + grid_size] + y[jnext + grid_size] - 4 * y[gid + grid_size])
        + g(y[gid], y[gid + grid_size], dc_par);
}


template <typename value_type, typename index_type>
__device__ __inline__ void eval_model_rhs(
    const value_type* y,
    value_type& rhs_u,
    value_type& rhs_v,
    const index_type gid,
    const index_type grid_resolution,
    const index_type grid_size
)
{
    auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
    auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
    auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
    auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

    value_type u = y[gid];
    value_type v = y[gid + grid_size];

    rhs_u = dc_nu1_hinv2 * (y[iprev] + y[inext] + y[jprev] + y[jnext] - 4 * u)
        + f(u, v, dc_par);

    rhs_v = dc_nu2_hinv2 * (y[iprev + grid_size] + y[inext + grid_size] + y[jprev + grid_size] + y[jnext + grid_size] - 4 * v)
        + g(u, v, dc_par);
}

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

template <typename value_type, typename index_type>
__global__ void prev_state_overwrite_stage(
    value_type* t_sys_state,
    const index_type* t_run_indices,
    const index_type t_sys_size,
    const index_type sim_number
)
{
    auto tid = threadIdx.x;
    auto fid = blockIdx.x * blockDim.x + tid;
    auto vid = fid / t_sys_size;
    auto gid = fid % t_sys_size;
    if (fid < t_sys_size * sim_number)
    {
        value_type* y_start = t_sys_state + 4 * t_sys_size * t_run_indices[2 * vid];
        value_type* y_prev = y_start + t_sys_size;

        auto counter = t_run_indices[2 * vid + 1];
        if (counter < 2)
            counter = 2;
        auto y_shift = 2 - (counter) % 3;
        value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * t_sys_size;
       
        y_prev[gid] = y[gid];
    }
}

template <typename value_type, typename index_type>
__global__ void update_time_step_data(
    const value_type* t_temp_data,
    value_type* t_time_step_data,
    const index_type* t_run_indices,
    const index_type t_sim_number
)
{
    auto mid = blockIdx.x * blockDim.x + threadIdx.x;
    if (mid < t_sim_number)
    {
        auto sid = t_run_indices[3 * mid];
        t_time_step_data[sid] = t_temp_data[mid];
    }
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

// y=temp1*f(yjm1)+temp2*yjm1+temp3*yjm2;
/*
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
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed
    auto gid = fid % grid_size; // index of the node in spatial grid, which is being processed
    auto sys_size = 2 * grid_size;

    if (mid < sim_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        auto sid = t_run_indices[3 * mid];
        auto pos_recf = t_run_indices[3 * mid + 1];
        auto counter = t_run_indices[3 * mid + 2];

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
        value_type rhs_u, rhs_v;
        //eval_model_rhs(yjm1, rhs_u, rhs_v, gid, grid_resolution, grid_size);
        //{
            auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
            auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
            auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
            auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

            value_type u = yjm1[gid];
            value_type v = yjm1[gid + grid_size];

            rhs_u = dc_nu1_hinv2 * (yjm1[iprev] + yjm1[inext] + yjm1[jprev] + yjm1[jnext] - 4 * u)
                + f(u, v);

            rhs_v = dc_nu2_hinv2 * (yjm1[iprev + grid_size] + yjm1[inext + grid_size] + yjm1[jprev + grid_size] + yjm1[jnext + grid_size] - 4 * v)
                + g(u, v);
        //}

        y[gid] = temp1 * rhs_u + temp2 * u + temp3 * yjm2[gid];
        y[gid + grid_size] = temp1 * rhs_v + temp2 * v + temp3 * yjm2[gid + grid_size];
    }
}
*/

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

/*
// grid: (block index i, block index j, model index)
// block: (thread index i, thread index j)
template <typename value_type, typename index_type>
__global__ void r2_rec_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    extern __shared__ value_type sm[];

    auto grid_size = grid_resolution * grid_resolution;
    auto gid = grid_resolution * (blockDim.y * blockIdx.y + threadIdx.y) + blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_i = blockDim.x * blockIdx.x + threadIdx.x;
    auto gid_j = blockDim.y * blockIdx.y + threadIdx.y;
    auto sys_size = 2 * grid_size;

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

    
    // model r.h.s. evaluation
    // first we load the block data into the shared memory, which
    // has the following 2D structure: 2 * (blockDim.x, blockDim.y)
    auto smid = blockDim.x * threadIdx.y + threadIdx.x;
    auto sm_grid_size = blockDim.x * blockDim.y;

    sm[smid] = yjm1[gid];
    sm[smid + sm_grid_size] = yjm1[gid + grid_size];

    __syncthreads();

    value_type* u_prev_i;
    value_type* u_prev_j;
    value_type* u_next_i;
    value_type* u_next_j;

    value_type* v_prev_i;
    value_type* v_prev_j;
    value_type* v_next_i;
    value_type* v_next_j;

    if ((threadIdx.x == 0) && (gid_i > 0))
    {
        u_prev_i = &yjm1[gid - 1];
        v_prev_i = u_prev_i + grid_size;
    }
    else
    {
        u_prev_i = &sm[smid - (gid_i > 0)];
        v_prev_i = u_prev_i + sm_grid_size;
    }

    if ((threadIdx.x == blockDim.x - 1) && (gid_i < grid_resolution - 1))
    {
        u_next_i = &yjm1[gid + 1];
        v_next_i = u_next_i + grid_size;
    }
    else
    {
        u_next_i = &sm[smid + (gid_i < grid_resolution - 1)];
        v_next_i = u_next_i + sm_grid_size;
    }

    if ((threadIdx.y == 0) && (gid_j > 0))
    {
        u_prev_j = &yjm1[gid - grid_resolution];
        v_prev_j = u_prev_j + grid_size;
    }
    else
    {
        u_prev_j = &sm[smid - blockDim.x * (gid_j > 0)];
        v_prev_j = u_prev_j + sm_grid_size;
    }
         
    if ((threadIdx.y == blockDim.y - 1) && (gid_j < grid_resolution - 1))
    {
        u_next_j = &yjm1[gid + grid_resolution];
        v_next_j = u_next_j + grid_size;
    }
    else
    {
        u_next_j = &sm[smid + blockDim.x * (gid_j < grid_resolution - 1)];
        v_next_j = u_next_j + sm_grid_size;
    }

    // y=temp1*f(yjm1)+temp2*yjm1+temp3*yjm2;
    value_type rhs_u, rhs_v;
    rhs_u = dc_nu1_hinv2 * (
        *u_prev_i 
        + *u_prev_j
        + *u_next_i
        + *u_next_j
        - 4 * sm[smid])
        + f(sm[smid], sm[smid + sm_grid_size]);

    
    rhs_v = dc_nu2_hinv2 * (
        *v_prev_i
        + *v_prev_j
        + *v_next_i
        + *v_next_j
        - 4 * sm[smid + sm_grid_size])
        + g(sm[smid], sm[smid + sm_grid_size]);

    
    //value_type rhs_test_u, rhs_test_v;
    //eval_model_rhs(yjm1, rhs_test_u, rhs_test_v, gid, grid_resolution, grid_size);

    //if (abs(rhs_v - rhs_test_v) > 1)
    //{
    //    printf("tid_i: %i, tid_j: %i, gid_i: %i, gid_j: %i, smid: %i \n",
    //       threadIdx.x, threadIdx.y, gid_i, gid_j, smid);
    //}
    

    y[gid] = temp1 * rhs_u + temp2 * sm[smid] + temp3 * yjm2[gid];
    y[gid + grid_size] = temp1 * rhs_v + temp2 * sm[smid + sm_grid_size] + temp3 * yjm2[gid + grid_size];
    
}
*/

/*
template <typename value_type, typename index_type>
__global__ void r2_rec_stage(
    value_type* t_sys_state,
    value_type* t_time_steps,
    const index_type* t_run_indices,
    const unsigned int grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution; // full size of the spatial grid (one component)
    auto sys_size = 2 * grid_size; // full size of the system (two components)
    
    auto fid = blockIdx.x * blockDim.x + threadIdx.x; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed; range: [0, sim_number)
    auto lid = fid % grid_size; // index of the current thread with respect to the current model; range: [0,grid_size)
    auto wid = lid / warpSize; // index of the current warp; range: [0, grid_size / warp_size)
    auto lane = threadIdx.x % warpSize;
    
    // determining the position of the current thread with respect to warp and spatial grid
    auto offset_i = (8 * wid) % grid_resolution;
    auto offset_j = 4 * ((8 * wid) / grid_resolution);
    auto lane_i = lane % 8;
    auto lane_j = lane / 8;
    auto gid = grid_resolution * (offset_j + lane_j) + offset_i + lane_i;
    auto gid_i = gid % grid_resolution;
    auto gid_j = gid / grid_resolution;
    auto grid_coeff = (gid_i > 0) + (gid_i < grid_resolution - 1) + (gid_j > 0) + (gid_j < grid_resolution - 1);
    
    auto sid = t_run_indices[3 * mid];
    auto pos_recf = t_run_indices[3 * mid + 1];
    auto counter = t_run_indices[3 * mid + 2];

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
    value_type u = yjm1[gid];
    value_type v = yjm1[gid + grid_size];

    value_type rhs_u = -grid_coeff * u;
    value_type rhs_v = -grid_coeff * v;

    // diffusion operator
    // right
    rhs_u += (lane_j < 3) * __shfl_down_sync(0xffffffff, u, 8);
    rhs_v += (lane_j < 3) * __shfl_down_sync(0xffffffff, v, 8);
    if ((lane_j == 3) && (gid_j < grid_resolution - 1))
    {
        rhs_u += yjm1[gid + grid_resolution];
        rhs_v += yjm1[grid_size + gid + grid_resolution];
    }

    // left
    rhs_u += (lane_j > 0) * __shfl_up_sync(0xffffffff, u, 8);
    rhs_v += (lane_j > 0) * __shfl_up_sync(0xffffffff, v, 8);
    if ((lane_j == 0) && (gid_j > 0))
    {
        rhs_u += yjm1[gid - grid_resolution];
        rhs_v += yjm1[grid_size + gid - grid_resolution];
    }

    // top
    rhs_u += (lane_i < 7) * __shfl_down_sync(0xffffffff, u, 1);
    rhs_v += (lane_i < 7) * __shfl_down_sync(0xffffffff, v, 1);
    if ((lane_i == 7) && (gid_i < grid_resolution - 1))
    {
        rhs_u += yjm1[gid + 1];
        rhs_v += yjm1[grid_size + gid + 1];
    }

    // bottom
    rhs_u  += (lane_i > 0) * __shfl_up_sync(0xffffffff, u, 1);
    rhs_v  += (lane_i > 0) * __shfl_up_sync(0xffffffff, v, 1);
    if ((lane_i == 0) && (gid_i > 0))
    {
        rhs_u += yjm1[gid - 1];
        rhs_v += yjm1[grid_size + gid - 1];
    }

    //if (wid == 31)
    //  printf("tid: %i, warp: %i, lane_i: %i, lane_j:%i, gid_i: %i, gid_j: %i, gid: %i, grid_coeff: %i \n", 
    //      fid, wid, lane_i, lane_j, gid_i, gid_j, gid, grid_coeff);
  
    // reaction term
    rhs_u = dc_nu1_hinv2 * rhs_u + f(u, v);
    rhs_v = dc_nu2_hinv2 * rhs_v + g(u, v);

    y[gid] = temp1 * rhs_u + temp2 * u + temp3 * yjm2[gid];
    y[gid + grid_size] = temp1 * rhs_v + temp2 * v + temp3 * yjm2[gid + grid_size];
}
*/


// yjm2=yjm1; yjm1 = y;
template <typename value_type, typename index_type>
__global__ void r2_rec_shift_stage(
    value_type* t_sys_state,
    const index_type* t_run_indices,
    const index_type t_sys_size,
    const index_type sim_number
)
{
    auto tid = threadIdx.x;
    auto fid = blockIdx.x * blockDim.x + tid;
    auto vid = fid / t_sys_size;
    auto gid = fid % t_sys_size;
    if (fid < t_sys_size * sim_number)
    {
        value_type* y = t_sys_state + 4 * t_sys_size * t_run_indices[vid];
        value_type* yjm1 = y + 2 * t_sys_size;
        value_type* yjm2 = yjm1 + t_sys_size;

        yjm2[gid] = yjm1[gid];
        yjm1[gid] = y[gid];
    }
}

// yjm1 = y + temp1 * f(neqn, y);
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

    auto sid = t_run_indices[2 * blockIdx.z];
    auto pos_fp = t_run_indices[2 * blockIdx.z + 1];
    auto counter = dc_ms[pos_fp];
    if (counter < 2)
        counter = 2;

    value_type* y_start = t_sys_state + 4 * sys_size * sid;

    auto y_shift = 2 - (counter) % 3;
    auto yjm1_shift = 2 - (counter - 1) % 3;

    value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * sys_size;
    value_type* yjm1 = y_start + (yjm1_shift > 0) * (yjm1_shift + 1) * sys_size;

    value_type time_step = t_time_steps[sid];

    value_type temp1 = time_step * dc_fp1[pos_fp];

    // yjm1 = y + temp1 * f(neqn, y);
    value_type rhs_u, rhs_v, u, v;
    eval_model_rhs(y, rhs_u, rhs_v, u, v, gid, gid_i, gid_j, grid_resolution, grid_size);

    yjm1[gid] = temp1 * rhs_u + u;
    yjm1[gid + grid_size] = temp1 * rhs_v + v;
    
}

// yjm1 = y + temp1 * f(neqn, y);
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

    value_type temp3_u = temp2 * (y[gid] - yjm2[gid]);
    value_type temp3_v = temp2 * (y[gid + grid_size] - yjm2[gid + grid_size]);

    // y = yjm1 + temp1*f(neqn,yjm1) + temp2*(y-yjm2);
    value_type rhs_u, rhs_v, u, v;
    eval_model_rhs(yjm1, rhs_u, rhs_v, u, v, gid, gid_i, gid_j, grid_resolution, grid_size);

    y[gid] = temp1 * rhs_u + u + temp3_u;
    y[gid + grid_size] = temp1 * rhs_v + v + temp3_v;
    
}


// t_ci1 = max(abs(y),abs(yn))*rto; err = sum((temp2*(y-yjm2)./(ato+t_ci1)).^2); err=sqrt(err/neqn);
template <typename value_type, typename index_type>
__global__ void local_error_est_stage(
    const value_type* t_sys_state,
    const value_type* t_time_steps,
    value_type* t_error,
    const index_type* t_run_indices,
    const index_type grid_resolution, // grid resolution
    const index_type sim_number // number of simulations
)
{
    static __shared__ value_type shared_mem[32];

    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto mid = blockIdx.x; // index of model which is processed
    auto sys_size = 2 * grid_size;
    value_type val = 0;

    auto sid = t_run_indices[2 * mid];
    auto pos_fp = t_run_indices[2 * mid + 1];
    auto counter = dc_ms[pos_fp];
    if (counter < 2)
        counter = 2;
    
    const value_type* y_start = t_sys_state + 4 * sys_size * sid;
    const value_type* y_prev = y_start + sys_size;

    auto y_shift = 2 - (counter) % 3;
    auto yjm2_shift = 2 - (counter - 2) % 3;

    const value_type* y = y_start + (y_shift > 0) * (y_shift + 1) * sys_size;
    const value_type* yjm2 = y_start + (yjm2_shift > 0) * (yjm2_shift + 1) * sys_size;

    value_type time_step = t_time_steps[sid];

    value_type temp2 = time_step * dc_fp2[pos_fp];
    
    auto gid = tid;
    while (gid < grid_size)
    {
        // t_ci1 = max(abs(y),abs(yn))*rto; err = sum((temp2*(y-yjm2)./(ato+t_ci1)).^2);
        value_type val_u = temp2 * (y[gid] - yjm2[gid]) / (dc_atol + max(abs(y[gid]), abs(y_prev[gid])) * dc_rtol);
        value_type val_v = 
            temp2 * (y[gid + grid_size] - yjm2[gid + grid_size]) / (dc_atol + max(abs(y[gid + grid_size]), abs(y_prev[gid + grid_size])) * dc_rtol);

        val += val_u * val_u;// +val_v * val_v;

        gid += blockDim.x;
    } // while
   
    __syncthreads();

    // reduction

    // indexes for the current thread
    const unsigned int lane = tid % warpSize;
    const unsigned int wid = tid / warpSize;

    // warp-level reduction
    val = warp_reduce_sum(val);

    // writing reduced values to shared memory
    if (lane == 0)
        shared_mem[wid] = val;

    __syncthreads();

    //read from shared memory only if that warp existed
    val = (tid < blockDim.x / warpSize) ? shared_mem[lane] : 0;

    if (wid == 0)
        val = warp_reduce_sum(val); //Final reduce within first warp

   
    if (tid == 0)
    {
        t_error[mid] = sqrt(val / sys_size);
    }
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















/*

template <typename value_type> 
__global__ void scalar_sum3(
    const value_type* w1, 
    const value_type* w2, 
    const value_type* w3, 
    value_type* w4, 
    const value_type* alpha1, 
    const value_type* alpha2, 
    const value_type* alpha3, 
    const unsigned int vector_size,
    const unsigned int sim_number
)
{
    auto tid = threadIdx.x;
    auto fid = blockIdx.x * blockDim.x + tid;
    auto vid = fid / vector_size;
    if (fid < vector_size * sim_number)
    {
        w4[fid] = alpha1[vid] * w1[fid] + alpha2[vid] * w2[fid] + alpha3[vid] * w3[fid];
    }
}

template <typename value_type> 
__global__ void eval_model_rhs(
    const value_type* y_global, // system state (sim_number * 2 * grid_resolution * grid_resolution)
    value_type* rhs_global, // system r.h.s. (sim_number * 2 * grid_resolution * grid_resolution)
    const unsigned int grid_resolution, // grid resolution
    const unsigned int sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed
    auto gid = fid % grid_size; // index of the node in spatial grid, which is being processed

    if (mid < sim_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        const value_type* y = y_global + 2 * mid * grid_size; // finding the system state for current model
        value_type* rhs = rhs_global + 2 * mid * grid_size; // finding the r.h.s. state for current model

        auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
        auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
        auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
        auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

        rhs[gid] = dc_par.nu1_hinv2 * (y[iprev] + y[inext] + y[jprev] + y[jnext] - 4 * y[gid])
            + f(y[gid], y[gid + grid_size]);

        rhs[gid + grid_size] = dc_par.nu2_hinv2 * (y[iprev + grid_size] + y[inext + grid_size] + y[jprev + grid_size] + y[jnext + grid_size] - 4 * y[gid + grid_size])
            + g(y[gid], y[gid + grid_size]);
    }
}

template <typename value_type>
__global__ void eval_model_rhs(
    const value_type* y_global, // system state (sim_number * 2 * grid_resolution * grid_resolution)
    value_type* rhs_global, // system r.h.s. (sim_number * 2 * grid_resolution * grid_resolution)
    const unsigned int* grid_indices, // pre-computed spatial grid indices in global array
    const unsigned int grid_resolution, // grid resolution
    const unsigned int total_element_number // total number of threads
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)

    if (fid < total_element_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        auto iprev = grid_indices[4 * fid];
        auto inext = grid_indices[4 * fid + 1];
        auto jprev = grid_indices[4 * fid + 2];
        auto jnext = grid_indices[4 * fid + 3];

        rhs_global[fid] = dc_par.nu1_hinv2 * (y_global[iprev] + y_global[inext] + y_global[jprev] + y_global[jnext] - 4 * y_global[fid])
            + f(y_global[fid], y_global[fid + grid_size]);

        rhs_global[fid + grid_size] = dc_par.nu2_hinv2 * (y_global[iprev + grid_size] + y_global[inext + grid_size] + y_global[jprev + grid_size] + y_global[jnext + grid_size] - 4 * y_global[fid + grid_size])
            + g(y_global[fid], y_global[fid + grid_size]);
    }
}

template <typename value_type>
__global__ void eval_linear_disk_part(
    value_type* y_lin_part_global, // Gersgorin disk information, len(system state)
    const unsigned int grid_resolution, // grid resolution
    const unsigned int sim_number // number of simulations
)
{
    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed
    auto gid = fid % grid_size; // index of the node in spatial grid, which is being processed

    if (mid < sim_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        value_type* y_lin_part = y_lin_part_global + 2 * mid * grid_size; // finding the r.h.s. state for current model

        auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
        auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
        auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
        auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

        int corr_coeff = (iprev == gid) + (inext == gid) + (jprev == gid) + (jnext == gid);

        y_lin_part[gid] = dc_par.nu1_hinv2 * (-4 + corr_coeff);
        y_lin_part[gid + grid_size] = dc_par.nu2_hinv2 * (-4 + corr_coeff);
    }
}

template <typename value_type>
__global__ void estimate_spectral_radius(
    const value_type* y_global,
    value_type* rho_global,
    const unsigned int grid_resolution, // grid resolution
    const unsigned int sim_number // number of simulations
)
{
    static __shared__ value_type shared_mem[32];

    auto grid_size = grid_resolution * grid_resolution;
    auto tid = threadIdx.x; // thread index (note that each thread processes TWO elements of the system)
    auto fid = blockIdx.x * blockDim.x + tid; // index of the current thread in global scope (not related yet to grid)
    auto mid = fid / grid_size; // index of model which is processed
    auto gid = fid % grid_size; // index of the node in spatial grid, which is being processed
    value_type val = 0;

    if (mid < sim_number) // actual number of blocks does not necessarily correspond to the grid resolution
    {
        const value_type* y = y_global + 2 * mid * grid_size; // finding the system state for current model

        gid = tid;
        while (gid < grid_size)
        {
            auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
            auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
            auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
            auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

            unsigned int lin_coeff = 4 - (iprev == gid) - (inext == gid) - (jprev == gid) - (jnext == gid);

            auto center_u = -dc_par.nu1_hinv2 * lin_coeff + dfdu(y[gid], y[gid + grid_size]);
            auto center_v = -dc_par.nu2_hinv2 * lin_coeff + dgdv(y[gid], y[gid + grid_size]);
            auto radius_u = dc_par.nu1_hinv2 * lin_coeff + abs(dfdv(y[gid], y[gid + grid_size]));
            auto radius_v = dc_par.nu2_hinv2 * lin_coeff + abs(dgdu(y[gid], y[gid + grid_size]));
            auto res_gid = max(abs(center_u) + radius_u, abs(center_v) + radius_v);
            val = max(val, res_gid);
    
            gid += blockDim.x;
        } // while
    }
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
        rho_global[mid] = val;
    }
}


template <typename value_type> __global__ void rock2step_shared(
    const value_type* y_prev_global,
    value_type* y_next_global,
    value_type* error,
    value_type* k1_global,
    value_type* k2_global,
    const value_type dt,
    const unsigned int grid_resolution,
    const unsigned int mz,
    const unsigned int mr,
    const value_type abs_tol,
    const value_type rel_tol
)
{
    extern __shared__ value_type k3[]; // sys_size * sizeof(value_type)

    auto tid = threadIdx.x;
    auto gid = 0u; // grid position
    auto grid_size = grid_resolution * grid_resolution;
    auto offset_global = 2 * blockIdx.x * grid_size;

    value_type* k1 = k1_global + offset_global;
    value_type* k2 = k2_global + offset_global;

    // MATLAB: fn = f(neqn, t, yn, fn); yjm2 = yn; yjm1 = yn + h * recf(mr) * fn;
    // CUDA: k1 = k2 = k3 = yn; k1 += dt*recf[mr-1] * f(k3);
    gid = tid;
    while (gid < grid_size)
    {
        k3[gid] = k2[gid] = k1[gid] = y_prev_global[offset_global + gid];
        k3[gid + grid_size] = k2[gid + grid_size] = k1[gid + grid_size] = y_prev_global[offset_global + gid + grid_size];
        gid += blockDim.x;
    } // while
    __syncthreads();

    // MODEL RHS EVALUATION
    gid = tid;
    while (gid < grid_size)
    {
        auto iprev = (gid < grid_resolution) * gid + (gid >= grid_resolution) * (gid - grid_resolution);
        auto inext = (gid + grid_resolution < grid_size) * (gid + grid_resolution) + (gid + grid_resolution >= grid_size) * gid;
        auto jprev = (gid % grid_resolution == 0) * gid + (gid % grid_resolution > 0) * (gid - 1);
        auto jnext = (gid % grid_resolution == grid_resolution - 1) * gid + (gid % grid_resolution < grid_resolution - 1) * (gid + 1);

        k1[gid] += dt * dc_recf[mr - 1] * (
            dc_par.nu1_hinv2 * (k3[iprev] + k3[inext] + k3[jprev] + k3[jnext] - 4 * k3[gid])
            + f(k3[gid], k3[gid + grid_size])
            );

        k1[gid + grid_size] += dt * dc_recf[mr - 1] * (
            dc_par.nu2_hinv2 * (k3[iprev + grid_size] + k3[inext + grid_size] + k3[jprev + grid_size] + k3[jnext + grid_size] - 4 * k3[gid + grid_size])
            + g(k3[gid], k3[gid + grid_size])
            );
        gid += blockDim.x;
    }
    __syncthreads();

    // saving the result to global memory
    gid = tid;
    while (gid < grid_size)
    {
        y_next_global[offset_global + gid] = k1[gid];
        y_next_global[offset_global + gid + grid_size] = k1[gid + grid_size];
        gid += blockDim.x;
    } // while
}

*/