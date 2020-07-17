#pragma once

#include <spice/util/random.h>


namespace spice
{
struct backend
{
	explicit backend( unsigned long long seed )
	    : rng( seed )
	{
		spice_assert( seed > 0 );
	}

	template <typename T>
	static void atomic_add( T & var, T val )
	{
		var += val;
	}

	// @return random no. in [0, 1)
	float rand() { return util::uniform_left_inc( rng ); }

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
	util::xoroshiro128p rng;
};
} // namespace spice
