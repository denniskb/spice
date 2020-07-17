#pragma once

#include <limits>
#include <type_traits>
#include <typeinfo>


namespace spice
{
namespace util
{
#pragma warning( push )
#pragma warning( disable : 26472 ) // "Don't use static_cast for arithmetic conversion" (the whole
                                   // point is to provide an explicit narrowing cast so the
                                   // programmer can communicate his intent)
template <typename To, typename From>
To narrow_cast( From x )
{
	return static_cast<To>( x );
}
#pragma warning( pop )

/**
 * Disable
 * - 4018 "signed/unsigned mismatch" (integer promotion rules actually do the right thing here
 *                                   given our prior checks)
 * - 4100 "unreferenced formal parameter" (code removed via constexpr if)
 * - 4702 "unreachable code" (code removed via constexpr if)
 * - 26472 "Don't use static_cast for arithmetic conversion" (we have asserted that no narrowing can
 *                                                            occur, static_cast is actually what we
 *                                                            want here)
 */
#pragma warning( push )
#pragma warning( disable : 4018 4100 4702 )

/**
 * Attempts to cast one integral type (singed or unsigned) into another (signed or unsigned).
 * If @x can be represented as an @To, this function is equivalent to static_cast<@To>(@x).
 * Otherwise throws a std::bad_cast.
 */
// @contract To, From integral types (already asserted)
template <typename To, typename From>
To narrow_int( From x )
{
	static_assert(
	    std::is_integral_v<To> && std::is_integral_v<From>,
	    "narrow_int() can only be performed on integral types" );

	// narrow_int<T, T>()
	if constexpr( std::is_same_v<From, To> ) return x;

	// signed -> unsigned
	if constexpr( std::is_signed_v<From> && std::is_unsigned_v<To> )
		if( x < From( 0 ) ) throw std::bad_cast();

	// fast-path:
	// signed   -> signed
	// signed+  -> unsigned
	// unsigned -> unsigned
	// unsigned -> signed
	if constexpr( std::numeric_limits<To>::max() >= std::numeric_limits<From>::max() )
		return narrow_cast<To>( x );

	// value inspection:
	// signed  -> signed
	// signed+ -> unsigned
	if constexpr( std::is_signed_v<From> )
	{
		if( x <= std::numeric_limits<To>::max() && x >= std::numeric_limits<To>::min() )
			return narrow_cast<To>( x );
	}
	// unsigned -> unsigned
	// unsigned -> signed
	else
	{
		if( x <= std::numeric_limits<To>::max() ) return narrow_cast<To>( x );
	}

	throw std::bad_cast();
}

#pragma warning( pop )
} // namespace util
} // namespace spice
