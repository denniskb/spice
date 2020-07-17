#pragma once

#include <spice/util/assert.h>
#include <spice/util/host_defines.h>
#include <spice/util/type_traits.h>

#include <cmath>


namespace spice
{
namespace util
{
HYBRID inline unsigned rotl32( unsigned x, unsigned k ) { return ( x << k ) | ( x >> ( 32 - k ) ); }
HYBRID
inline unsigned long long rotl64( unsigned long long x, unsigned long long k )
{
	return ( x << k ) | ( x >> ( 64 - k ) );
}
HYBRID inline unsigned long long hash( unsigned long long x )
{
	spice_assert( x > 0 );

	x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9llu;
	x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebllu;
	x = x ^ ( x >> 31 );

	return x;
}
#ifndef __CUDA_ARCH__
inline float sinpif( float x ) { return std::sin( 3.14159265359f * x ); }
#endif

// http://prng.di.unimi.it/xoshiro128plus.c
class xoroshiro128p
{
public:
	using result_type = unsigned;
	constexpr unsigned min() { return 0; }
	constexpr unsigned max() { return UINT_MAX; }

	HYBRID inline explicit xoroshiro128p( unsigned long long seed )
	{
		spice_assert( seed > 0 );

		auto h = hash( seed );
		s0 = (unsigned)h;
		s1 = (unsigned)( h >> 32 );

		h = hash( h );
		s2 = (unsigned)h;
		s3 = (unsigned)( h >> 32 );
	}

	HYBRID inline unsigned operator()()
	{
		auto result = s0 + s3;

		auto t = s1 << 9;

		s2 ^= s0;
		s3 ^= s1;
		s1 ^= s2;
		s0 ^= s3;

		s2 ^= t;

		s3 = rotl32( s3, 11 );

		return result;
	}

private:
	unsigned s0, s1, s2, s3;
};

// http://prng.di.unimi.it/xoshiro256starstar.c
class xoroshiro256ss
{
public:
	using result_type = unsigned long long;
	constexpr result_type min() { return 0; }
	constexpr result_type max() { return ULLONG_MAX; }

	HYBRID inline explicit xoroshiro256ss( unsigned long long seed )
	{
		spice_assert( seed > 0 );

		s0 = hash( seed );
		s1 = hash( s0 );
		s2 = hash( s1 );
		s3 = hash( s2 );
	}

	HYBRID inline result_type operator()()
	{
		auto result = rotl64( s1 * 5, 7 ) * 9;

		auto t = s1 << 17;

		s2 ^= s0;
		s3 ^= s1;
		s1 ^= s2;
		s0 ^= s3;

		s2 ^= t;

		s3 = rotl64( s3, 45 );

		return result;
	}

private:
	unsigned long long s0, s1, s2, s3;
};


// @return rand no. in [0, 1)
template <typename Gen>
HYBRID float uniform_left_inc( Gen & gen )
{
	return ( (unsigned)gen() >> 8 ) / 16777216.0f;
}

// @return rand no. in (0, 1]
template <typename Gen>
HYBRID float uniform_right_inc( Gen & gen )
{
	return ( ( (unsigned)gen() >> 8 ) + 1.0f ) / 16777216.0f;
}

template <typename Gen>
HYBRID float exprnd( Gen & gen )
{
	return -logf( uniform_right_inc( gen ) );
}

// @param m mu
// @param s sigma (standard deviation)
template <typename Gen>
HYBRID static float normrnd( Gen & gen, float m = 0.0f, float s = 1.0f )
{
	return fmaf(
	    sqrtf( -2 * logf( uniform_right_inc( gen ) ) ) * sinpif( 2 * uniform_left_inc( gen ) ),
	    s,
	    m );
}

template <typename Gen>
HYBRID int binornd( Gen & gen, int N, float p )
{
#ifndef __CUDA_ARCH__
	using std::max;
	using std::min;
#endif

	return min( N, max( 0, (int)lrintf( normrnd( gen, N * p, sqrtf( N * p * ( 1 - p ) ) ) ) ) );
}
} // namespace util
} // namespace spice
