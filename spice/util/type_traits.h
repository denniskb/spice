#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <typeinfo>


namespace spice
{
namespace util
{
template <typename To, typename _From>
To narrow_cast( _From x )
{
	return static_cast<To>( x );
}

// 4018 "signed/uint_ mismatch": integerpromotion does the right thing here
// 4702 "unreachable code": code removed via constexpr if
#pragma warning( push )
#pragma warning( disable : 4018 4702 )
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
/**
 * Attempts to convert one numerical type to another. If the destination *type* cannot
 * hold the source *value* without loss of precision, an exception is thrown.
 *
 * If the success of the conversion can be determined at compile time,
 * no runtime checks are performed whatsoever and 'narrow' degenerates to a static_cast.
 */
template <typename To, typename From>
constexpr To narrow( From x )
{
	using _From = std::remove_reference_t<std::remove_cv_t<From>>;

	static_assert(
	    std::is_same<To, std::remove_reference_t<std::remove_cv_t<To>>>::value,
	    "narrow() may only return value types" );
	static_assert(
	    std::is_arithmetic<_From>::value && std::is_arithmetic<To>::value,
	    "narrow() is inteded for arithmetic types only" );
	static_assert(
	    !std::is_same<_From, To>::value, "pointless conversion between identical types" );

	constexpr bool from_real = std::is_floating_point<_From>::value;
	constexpr bool from_int = std::is_integral<_From>::value;
	constexpr bool from_signed = std::is_signed<_From>::value;
	constexpr bool from_unsigned = std::is_unsigned<_From>::value;
	constexpr bool to_real = std::is_floating_point<To>::value;
	constexpr bool to_int = std::is_integral<To>::value;
	constexpr bool to_unsigned = std::is_unsigned<To>::value;
	constexpr auto from_size = std::numeric_limits<_From>::digits;
	constexpr auto to_size = std::numeric_limits<To>::digits;
	constexpr auto to_min = std::numeric_limits<To>::min();
	constexpr auto to_max = std::numeric_limits<To>::max();

	// int_ -> int_
	if constexpr( from_int && to_int )
	{
		// signed -> uint_
		if constexpr( from_signed && to_unsigned )
			if( x < 0 ) throw std::bad_cast();

		// signed -> signed
		// uint_ -> uint_
		// uint_ -> signed
		if constexpr( to_size >= from_size )
			return static_cast<To>( x );
		else if( ( from_unsigned || x >= to_min ) && x <= to_max )
			return static_cast<To>( x );
	}

	// int_ -> real
	if constexpr( from_int && to_real )
	{
		if constexpr( to_size >= from_size ) return static_cast<To>( x );

		std::make_unsigned_t<_From> y;
		if constexpr( from_unsigned )
			y = x;
		else
			y = std::abs( x );

		if( y <= 1llu << to_size ) return static_cast<To>( x );

		int_ a = std::numeric_limits<decltype( y )>::digits - 1;
		int_ b = 0;

		for( ; a >= 0; a-- )
			if( ( y >> a ) & 1 ) break;
		for( ; b <= a; b++ )
			if( ( y >> b ) & 1 ) break;

		if( a - b + 1 <= to_size ) return static_cast<To>( x );
	}

	// real -> int_
	if constexpr( from_real && to_int )
	{
		_From tmp;
		auto frac = std::modf( x, &tmp );

		if( frac == 0 && x >= to_min && x < std::exp2( to_size ) ) return static_cast<To>( x );
	}

	// real -> real
	if constexpr( from_real && to_real )
	{
		if constexpr( to_size >= from_size ) return static_cast<To>( x );
		if( x == static_cast<_From>( static_cast<To>( x ) ) ) return static_cast<To>( x );
	}

	throw std::bad_cast();
}
#pragma GCC diagnostic pop
#pragma warning( pop )
} // namespace util
} // namespace spice
