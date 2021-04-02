#pragma once

#include <spice/util/random.h>


namespace spice
{
namespace cuda
{
struct backend
{
	__device__ explicit backend( ulong_ seed )
	    : rng( seed )
	{
		spice_assert( seed > 0 );
	}

	template <typename T>
	__device__ static void atomic_add( T & var, T val )
	{
		atomicAdd( &var, val );
	}

	// @return rand float in [0, 1)
	__device__ float rand() { return util::uniform_left_inc( rng ); }

	template <typename T>
	__device__ static T min( T x, T hi )
	{
		return fminf( x, hi );
	}

	template <typename T>
	__device__ static T max( T lo, T x )
	{
		return fmaxf( lo, x );
	}

	template <typename T>
	__device__ static T clamp( T x, T lo, T hi )
	{
		return min( hi, max( lo, x ) );
	}

	template <typename T>
	__device__ static T exp( T x )
	{
		return expf( x );
	}

	__device__ static float pow( float x, float y ) { return powf( x, y ); }

private:
	spice::util::xoroshiro128p rng;
};
} // namespace cuda
} // namespace spice
