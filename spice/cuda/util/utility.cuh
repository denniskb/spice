#pragma once

#include <spice/cuda/util/defs.h>


#ifndef MASK_ALL
#define MASK_ALL 0xffffffff
#endif

namespace spice
{
namespace cuda
{
namespace util
{
__device__ __forceinline__ int threadid() { return threadIdx.x + blockIdx.x * blockDim.x; }
__device__ __forceinline__ int laneid() { return threadIdx.x % WARP_SZ; }
__device__ __forceinline__ int warpid_block() { return threadIdx.x / WARP_SZ; }
__device__ __forceinline__ int warpid_grid() { return threadid() / WARP_SZ; }
__device__ __forceinline__ int num_threads() { return blockDim.x * gridDim.x; }
__device__ __forceinline__ int num_warps() { return ( num_threads() + WARP_SZ - 1 ) / WARP_SZ; }
} // namespace util
} // namespace cuda
} // namespace spice
