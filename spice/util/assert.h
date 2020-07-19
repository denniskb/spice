#pragma once


namespace spice
{
namespace util
{
void _assert( char const * cond, char const * file, int line, char const * msg = nullptr );
}
} // namespace spice

#ifdef __CUDA_ARCH__
#if defined( SPICE_ASSERT_RELEASE ) || !defined( NDEBUG )
#define spice_assert( X ) \
	if( !( X ) ) asm( "trap;" );
#else
#define spice_assert( X )
#endif
#else
#if defined( SPICE_ASSERT_RELEASE ) || !defined( NDEBUG )
#define spice_assert( X, ... ) \
	(void)( ( X ) || ( ::spice::util::_assert( #X, __FILE__, __LINE__, ##__VA_ARGS__ ), 0 ) )
#else
#define spice_assert( X, ... )
#endif
#endif