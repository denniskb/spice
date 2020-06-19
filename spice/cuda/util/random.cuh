#pragma once

#include <cfloat>
#include <climits>


namespace spice
{
namespace cuda
{
namespace util
{
__device__ inline unsigned long long hash( unsigned long long x )
{
	x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9llu;
	x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebllu;
	x = x ^ ( x >> 31 );
	return x;
}

// RNG with two words of state and a period of 2^64-1
class xorshift64
{
public:
	__device__ explicit xorshift64( unsigned long long seed )
	{
		auto h = hash( seed + 1 );
		x = (unsigned)h | 1;
		y = (unsigned)( h >> 32 ) | 1;
	}

	__device__ __inline__ unsigned operator()()
	{
		unsigned t = x ^ ( x << 13 );
		x = y;
		return y = ( y ^ ( y >> 5 ) ) ^ ( t ^ ( t >> 17 ) );
	}

private:
	unsigned x, y;
};

#define rotl( x, k ) ( ( x << k ) | ( x >> ( 32 - k ) ) )
class xoroshiro64
{
public:
	__device__ explicit xoroshiro64( unsigned long long seed )
	{
		auto h = hash( seed + 1 );
		x = (unsigned)h | 1;
		y = (unsigned)( h >> 32 ) | 1;
	}

	__device__ __inline__ unsigned operator()()
	{
		unsigned result = x * 0x9E3779BB;

		y ^= x;
		x = rotl( x, 26 ) ^ y ^ ( y << 9 );
		y = rotl( y, 13 );

		return result;
	}

private:
	unsigned x, y;
};

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
