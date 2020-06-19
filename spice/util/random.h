#pragma once

#include <spice/util/assert.h>
#include <spice/util/type_traits.h>

#include <random>


namespace spice
{
namespace util
{
class xorshift64
{
public:
	using result_type = unsigned;

	explicit xorshift64( unsigned long long seed );

	unsigned operator()();

	static unsigned min();
	static unsigned max();

private:
	unsigned x, y;
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
