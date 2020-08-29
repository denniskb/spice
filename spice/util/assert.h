#pragma once

#include <spice/util/host_defines.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>


namespace spice
{
namespace util
{
namespace detail
{
inline char separator()
{
#ifdef _WIN32
	return '\\';
#else
	return '/';
#endif
}
} // namespace detail
inline HYBRID void
_assert( char const * cond, char const * file, int line, char const * msg = nullptr )
{
#ifdef __CUDA_ARCH__
	asm( "trap;" );
#else
	char buffer[1024];
	snprintf(
	    buffer,
	    sizeof( buffer ),
	    msg ? "[%s: %d] assertion '%s' failed: %s\n" : "[%s: %d] assertion '%s' failed\n",
	    strrchr( file, detail::separator() ) + 1,
	    line,
	    cond,
	    msg );

	throw std::invalid_argument( buffer );
#endif
}
} // namespace util
} // namespace spice


#if defined( SPICE_ASSERT_RELEASE ) || !defined( NDEBUG )
#define spice_assert( X, ... ) \
	(void)( ( X ) || ( ::spice::util::_assert( #X, __FILE__, __LINE__, ##__VA_ARGS__ ), 0 ) )
#else
#define spice_assert( X, ... )
#endif