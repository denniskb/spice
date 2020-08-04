#pragma once

#include <limits>
#include <type_traits>
#include <typeinfo>


namespace spice
{
namespace util
{
template <typename To, typename From>
To narrow_cast( From x )
{
	return static_cast<To>( x );
}

#pragma warning( push )
#pragma warning( disable : 4702 ) // "unreachable code" (code removed via constexpr if)
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
	static_assert(
	    !(std::is_reference_v<From> || std::is_reference_v<To>),
	    "please call narrow() with value types only" );
	static_assert(
	    std::is_arithmetic_v<From> && std::is_arithmetic_v<To>,
	    "narrow() is inteded for arithmetic types only" );
	static_assert(
	    !std::is_same_v<std::remove_cv<From>, std::remove_cv<To>>,
	    "pointless conversion between identical types" );

	constexpr bool from_real = std::is_floating_point_v<From>;
	constexpr bool from_int = std::is_integral_v<From>;
	constexpr bool from_signed = std::is_signed_v<From>;
	constexpr bool from_unsigned = std::is_unsigned_v<From>;
	constexpr bool to_real = std::is_floating_point_v<To>;
	constexpr bool to_int = std::is_integral_v<To>;
	constexpr bool to_unsigned = std::is_unsigned_v<To>;
	constexpr auto from_size = std::numeric_limits<From>::digits;
	constexpr auto to_size = std::numeric_limits<To>::digits;
	constexpr auto to_min = std::numeric_limits<To>::min();
	constexpr auto to_max = std::numeric_limits<To>::max();

	// int -> int
	if constexpr( from_int && to_int )
	{
		// signed -> unsigned
		if constexpr( from_signed && to_unsigned )
			if( x < 0 ) throw std::bad_cast();

		// signed -> signed
		// unsigned -> unsigned
		// unsigned -> signed
		if constexpr( to_size > from_size )
			return static_cast<To>( x );
		else if( ( from_unsigned || x >= to_min ) && x <= to_max )
			return static_cast<To>( x );
	}

	// * -> real
	// real -> *
	if constexpr( from_real || to_real )
	{
		// * -> real
		if constexpr( to_real && to_size >= from_size ) return static_cast<To>( x );
		// TODO: Handle int overflow!
		if( x == static_cast<From>( static_cast<To>( x ) ) ) return static_cast<To>( x );
	}

	throw std::bad_cast();
}
#pragma warning( pop )
} // namespace util
} // namespace spice
