#pragma once


namespace spice
{
namespace util
{
void _assert( char const * cond, char const * file, int line, char const * msg = nullptr );
} // namespace util
} // namespace spice


#if defined( SPICE_ASSERT_RELEASE ) || !defined( NDEBUG )
#ifndef __CUDA_ARCH__
#define spice_assert( X, ... ) \
	(void)( ( X ) || ( ::spice::util::_assert( #X, __FILE__, __LINE__, ##__VA_ARGS__ ), 0 ) )
#else
#define spice_assert( X, ... ) \
	if( !( X ) ) asm( "trap;" );
#endif
#else
#define spice_assert( X, ... )
#endif