#pragma once

#include <spice/cuda/util/defs.h>


#ifndef MASK_ALL
#define MASK_ALL 0xffffffffu
#endif

namespace spice
{
namespace cuda
{
namespace util
{
__device__ __forceinline__ int_ threadid() { return threadIdx.x + blockIdx.x * blockDim.x; }
__device__ __forceinline__ int_ laneid() { return threadIdx.x % WARP_SZ; }
__device__ __forceinline__ int_ warpid_block() { return threadIdx.x / WARP_SZ; }
__device__ __forceinline__ int_ warpid_grid() { return threadid() / WARP_SZ; }
__device__ __forceinline__ int_ num_threads() { return blockDim.x * gridDim.x; }
__device__ __forceinline__ int_ num_warps() { return ( num_threads() + WARP_SZ - 1 ) / WARP_SZ; }

inline __device__ uint_ active_mask( int_ i, int_ n )
{
	return MASK_ALL >> max( 0, ( i & 0xFFFFFFE0 ) + WARP_SZ - n );
}
} // namespace util
} // namespace cuda
} // namespace spice
