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
__device__ inline int_ threadid() { return threadIdx.x + blockIdx.x * blockDim.x; }
__device__ inline int_ laneid() { return threadIdx.x % WARP_SZ; }
__device__ inline int_ warpid_block() { return threadIdx.x / WARP_SZ; }
__device__ inline int_ warpid_grid() { return threadid() / WARP_SZ; }
__device__ inline int_ num_threads() { return blockDim.x * gridDim.x; }
__device__ inline int_ num_warps() { return ( num_threads() + WARP_SZ - 1 ) / WARP_SZ; }

// round down to next multiple of 32
__device__ inline int_ floor32( int_ x ) { return x & ~0x1f; }

// round up to next multiple of 32
__device__ inline int_ ceil32( int_ x ) { return floor32( x ) + 32; }

__device__ inline uint_ active_mask( int_ i, int_ n )
{
	static_assert( 32 == WARP_SZ, "TODO: Fix active_mask() logic (utility.cuh)" );
	spice_assert( ( ( i - laneid() ) & 0x1f ) == 0 );

	return MASK_ALL >> max( 0, ceil32( i ) - n );
}

__device__ inline float lerp( float a, float b, float weight_b )
{
	return a + weight_b * ( b - a );
}

__device__ inline uint_ set_lsbs( int_ n )
{
	spice_assert( n < 32 );
	return ( 1u << n ) - 1;
}
} // namespace util
} // namespace cuda
} // namespace spice
