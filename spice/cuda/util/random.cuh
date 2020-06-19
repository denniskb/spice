#pragma once

#include <cfloat>
#include <climits>


namespace spice
{
namespace cuda
{
namespace util
{
__device__ inline unsigned hash( unsigned x )
{
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( x >> 16 ) ^ x;

	return x;
}

// RNG with single word state and a period of 2^32-1
class xorshift
{
public:
	__device__ xorshift( unsigned seed = 1337 )
	    : a( hash( seed + 1 ) | 1 )
	{
	}

	__device__ __inline__ unsigned operator()()
	{
		unsigned x = a;
		x ^= x << 13;
		x ^= x >> 17;
		x ^= x << 5;
		return ( a = x );
	}

private:
	unsigned a;
};

// RNG with five words of state and a period of 2^160-2^32
class xorwow
{
public:
	__device__ xorwow( unsigned seed = 1337 )
	    : a( hash( seed + 1 ) | 1 )
	    , b( hash( hash( seed + 1 ) ) | 1 )
	    , c( hash( hash( hash( seed + 1 ) ) ) | 1 )
	    , d( hash( hash( hash( hash( seed + 1 ) ) ) ) | 1 )
	{
	}

	__device__ unsigned operator()()
	{
		unsigned t = d;
		unsigned s = a;

		d = c;
		c = b;
		b = s;

		t ^= t >> 2;
		t ^= t << 1;
		t ^= s ^ ( s << 4 );
		a = t;

		counter += 362437;

		return t + counter;
	}

	static __device__ __inline__ unsigned min() { return 0; }
	static __device__ __inline__ unsigned max() { return UINT_MAX; }

private:
	unsigned a;
	unsigned b;
	unsigned c;
	unsigned d;
	unsigned counter = 0;
};

class uniform_distr
{
public:
	// @return rand no. in [0, 1]
	template <typename Gen>
	static __device__ __inline__ float inclusive( Gen & gen )
	{
		return ( gen() & 0x00ffffff ) / 16777215.0f;
	}

	// @return rand no. in [0, 1)
	template <typename Gen>
	static __device__ __inline__ float left_inclusive( Gen & gen )
	{
		return ( gen() & 0x00ffffff ) / 16777216.0f;
	}

	// @return rand no. in (0, 1]
	template <typename Gen>
	static __device__ __inline__ float right_inclusive( Gen & gen )
	{
		// TODO: add 1 in int or float?
		return ( ( gen() & 0x00ffffff ) + 1.0f ) / 16777216.0f;
	}

	// @return rand no. in (0, 1)
	template <typename Gen>
	static __device__ __inline__ float exclusive( Gen & gen )
	{
		// TODO: "
		return ( ( gen() & 0x007fffff ) + 1.0f ) / 8388609.0f;
	}
};

class exp_distr
{
public:
	template <typename Gen>
	static __device__ __inline__ float rand( Gen & gen )
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
	static __device__ float rand( Gen & gen, float m = 0.0f, float s = 1.0f )
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
	static __device__ int rand( Gen & gen, int N, float p )
	{
		return min(
		    N,
		    max( 0, (int)lrintf( normal_distr::rand( gen, N * p, sqrtf( N * p * ( 1 - p ) ) ) ) ) );
	}
};
} // namespace util
} // namespace cuda
} // namespace spice
