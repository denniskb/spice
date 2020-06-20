#pragma once

#include <spice/util/assert.h>

#include <cfloat>
#include <climits>


namespace spice
{
namespace cuda
{
namespace util
{
class uniform_distr
{
public:
	// @return rand no. in [0, 1]
	template <typename Gen>
	__device__ __inline__ static float inclusive( Gen & gen )
	{
		return ( gen() >> 8 ) / 16777215.0f;
	}

	// @return rand no. in [0, 1)
	template <typename Gen>
	__device__ __inline__ static float left_inclusive( Gen & gen )
	{
		return ( gen() >> 8 ) / 16777216.0f;
	}

	// @return rand no. in (0, 1]
	template <typename Gen>
	__device__ __inline__ static float right_inclusive( Gen & gen )
	{
		// TODO: add 1 in int or float?
		return ( ( gen() >> 8 ) + 1.0f ) / 16777216.0f;
	}

	// @return rand no. in (0, 1)
	template <typename Gen>
	__device__ __inline__ static float exclusive( Gen & gen )
	{
		// TODO: "
		return ( ( gen() >> 9 ) + 1.0f ) / 8388609.0f;
	}
};

class exp_distr
{
public:
	template <typename Gen>
	__device__ __inline__ static float rand( Gen & gen )
	{
		return -logf( uniform_distr::right_inclusive( gen ) );
	}
};

class normal_distr
{
public:
	// @param m mu
	// @param s sigma (standard deviation)
	template <typename Gen>
	__device__ static float rand( Gen & gen, float m = 0.0f, float s = 1.0f )
	{
		return fmaf(
		    sqrtf( -2 * logf( uniform_distr::right_inclusive( gen ) ) ) *
		        sinpif( 2 * uniform_distr::left_inclusive( gen ) ),
		    s,
		    m );
	}
};

class binomial_distr
{
public:
	template <typename Gen>
	__device__ static int rand( Gen & gen, int N, float p )
	{
		return min(
		    N,
		    max( 0, (int)lrintf( normal_distr::rand( gen, N * p, sqrtf( N * p * ( 1 - p ) ) ) ) ) );
	}
};
} // namespace util
} // namespace cuda
} // namespace spice
