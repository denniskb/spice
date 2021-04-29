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

// round down to next multiple of 32
__device__ __forceinline__ int_ floor32( int_ x ) { return x & ~0x1f; }

// round up to next multiple of 32
__device__ __forceinline__ int_ ceil32( int_ x ) { return floor32( x ) + 32; }

__device__ __forceinline__ uint_ active_mask( int_ i, int_ n )
{
	static_assert( 32 == WARP_SZ, "TODO: Fix active_mask() logic (utility.cuh)" );
	spice_assert( ( ( i - laneid() ) & 0x1f ) == 0 );

	return MASK_ALL >> max( 0, ceil32( i ) - n );
}

__device__ __forceinline__ float lerp( float a, float b, float weight_b )
{
	return a + weight_b * ( b - a );
}
} // namespace util
} // namespace cuda
} // namespace spice
