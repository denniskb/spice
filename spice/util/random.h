#pragma once

#include <spice/util/assert.h>
#include <spice/util/host_defines.h>
#include <spice/util/if_constexpr.h>
#include <spice/util/type_traits.h>

#include <random>


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


template <typename Prec = double>
class exponential_distribution
{
public:
	static_assert( std::is_floating_point_v<Prec>, "Prec = precision = {float|double}" );

	using result_type = Prec;
	using param_type = Prec;

	exponential_distribution() = default;

	template <typename Gen>
	Prec operator()( Gen & gen )
	{
		std::conditional_t<std::is_same_v<Prec, float>, unsigned, unsigned long long> rnd = 0;

		for( int i = 0; i < std::numeric_limits<Prec>::digits;
		     i += std::numeric_limits<typename Gen::result_type>::digits )
			rnd |= gen() << i;

		return -std::log(
		    ( ( rnd >> 8 ) + 1 ) / static_cast<Prec>( 1llu << std::numeric_limits<Prec>::digits ) );
	}
};

#if defined( SPICE_ASSERT_RELEASE ) || !defined( NDEBUG )
template <typename Prec = int>
using binomial_distribution = std::binomial_distribution<Prec>;
#else
template <typename Prec = int>
class binomial_distribution
{
public:
	static_assert( std::is_integral_v<Prec>, "Prec = precision = (char, short, int, ....)" );

	using result_type = Prec;
	using param_type = Prec;

	explicit binomial_distribution( Prec n = 0, double p = 0.5 ) noexcept
	{
		spice_assert( n >= 0, "Invalid trial count" );
		spice_assert( p >= 0.0 && p <= 1.0, "Invalid probability" );
		param( n, p );
	}

	void reset() const {}
	Prec param() const { return _n; }
	double param2() const { return _p; }
	void param( Prec n, double p = 0.5 )
	{
		spice_assert( n >= 0, "Invalid trial count" );
		spice_assert( p >= 0.0 && p <= 1.0, "Invalid probability" );

		_n = n;
		_p = p;
		_d = {(float)( n * p ), (float)std::sqrt( n * p * ( 1.0 - p ) ) + FLT_EPSILON}; // for edge
		                                                                                // cases p =
		                                                                                // {0|1}
	}

	template <typename Gen>
	Prec operator()( Gen & g )
	{
		auto const result = narrow_cast<Prec>( std::round( _d( g ) ) );

		// If this invariant check fails often, consider using a true binomial distribution!
		spice_assert( result >= 0 && result <= _n, "binom. distr. approx. fail" );

		return std::max( 0, std::min( _n, result ) );
	}
	template <typename Gen>
	Prec operator()( Gen & g, Prec n, double p = 0.5 )
	{
		return binomial_distribution<Prec>( n, p )( g );
	}

	Prec min() const { return 0; }
	Prec max() const { return _n; }

private:
	Prec _n = 0;
	double _p = 0.5;
	std::normal_distribution<float> _d;
};

template <typename Prec>
bool operator==( binomial_distribution<Prec> const & a, binomial_distribution<Prec> const & b )
{
	return a.param() == b.param() && a.param2() == b.param2();
}
template <typename Prec>
bool operator!=( binomial_distribution<Prec> const & a, binomial_distribution<Prec> const & b )
{
	return a.param() != b.param() || a.param2() != b.param2();
}
#endif
} // namespace util
} // namespace spice
