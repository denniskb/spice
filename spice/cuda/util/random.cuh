#pragma once

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

class xorwow
{
public:
	__device__ xorwow( unsigned seed = 1337 )
	    : x( hash( seed + 1 ) )
	{
	}

	__device__ unsigned operator()()
	{
		unsigned t = ( x ^ ( x >> 2 ) );

		x = y;
		y = z;
		z = w;
		w = v;
		v = ( v ^ ( v << 4 ) ) ^ ( t ^ ( t << 1 ) );
		d += 362437u;

		return d + v;
	}

	static __device__ __inline__ unsigned min() { return 0; }
	static __device__ __inline__ unsigned max() { return UINT_MAX; }

private:
	unsigned x;
	unsigned y = 362436069u;
	unsigned z = 521288629u;
	unsigned w = 88675123u;
	unsigned v = 5783321u;
	unsigned d = 6615241u;
};

class uniform_distr
{
public:
	// @return rand no. in [0, 1]
	template <typename Gen>
	static __device__ __inline__ float inclusive( Gen & gen )
	{
		return gen() / (float)Gen::max();
	}

	// @return rand no. in (0, 1)
	template <typename Gen>
	static __device__ __inline__ float exclusive( Gen & gen )
	{
		return ( (float)gen() + 1 ) / ( (float)Gen::max() + 2 );
	}

	// @return rand no. in [0, 1)
	template <typename Gen>
	static __device__ __inline__ float left_inclusive( Gen & gen )
	{
		return gen() / ( (float)Gen::max() + 1 );
	}

	// @return rand no. in (0, 1]
	template <typename Gen>
	static __device__ __inline__ float right_inclusive( Gen & gen )
	{
		return ( (float)gen() + 1 ) / ( (float)Gen::max() + 1 );
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
		        cospif( 2 * uniform_distr::inclusive( gen ) ),
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
