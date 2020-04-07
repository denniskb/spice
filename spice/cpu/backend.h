#pragma once

#include <spice/util/random.h>


namespace spice
{
struct backend
{
	explicit backend( unsigned seed = 0 )
	    : rng( seed )
	{
	}

	template <typename T>
	static void atomic_add( T & var, T val )
	{
		var += val;
	}

	float rand() { return rng() / (float)rng.max(); }

	template <typename T>
	static T min( T x, T hi )
	{
		return std::min( x, hi );
	}

	template <typename T>
	static T max( T lo, T x )
	{
		return std::max( lo, x );
	}

	template <typename T>
	static T clamp( T x, T lo, T hi )
	{
		return min( hi, max( lo, x ) );
	}

	template <typename T>
	static T exp( T x )
	{
		return std::exp( x );
	}

private:
	util::xorwow rng;
};
} // namespace spice