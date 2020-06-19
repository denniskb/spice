#pragma once

#include <spice/cuda/util/random.cuh>


namespace spice
{
namespace cuda
{
struct backend
{
	__device__ explicit backend( unsigned long long seed )
	    : rng( seed )
	{
	}

	template <typename T>
	__device__ static void atomic_add( T & var, T val )
	{
		atomicAdd( &var, val );
	}

	// @return rand float in [0, 1)
	__device__ float rand() { return util::uniform_distr::left_inclusive( rng ); }

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

private:
	util::xoroshiro64 rng;
};
} // namespace cuda
} // namespace spice
