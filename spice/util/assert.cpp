#include <spice/util/assert.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>


static char separator()
{
#ifdef _WIN32
	return '\\';
#else
	return '/';
#endif
}


namespace spice
{
namespace util
{
void _assert( char const * cond, char const * file, int line, char const * msg /* = nullptr */ )
{
	char buffer[1024];
	snprintf(
	    buffer,
	    sizeof( buffer ),
	    msg ? "[%s: %d] assertion '%s' failed: %s\n" : "[%s: %d] assertion '%s' failed\n",
	    strrchr( file, separator() ) + 1,
	    line,
	    cond,
	    msg );

	throw std::invalid_argument( buffer );
}
} // namespace util
} // namespace spice
