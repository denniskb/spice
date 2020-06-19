#include "random.h"


static unsigned long long hash( unsigned long long x )
{
	x = ( x ^ ( x >> 30 ) ) * 0xbf58476d1ce4e5b9llu;
	x = ( x ^ ( x >> 27 ) ) * 0x94d049bb133111ebllu;
	x = x ^ ( x >> 31 );

	return x;
}


namespace spice::util
{
xorshift64::xorshift64( unsigned long long seed )
{
	auto h = hash( seed + 1 );
	x = (unsigned)h | 1;
	y = (unsigned)( h >> 32 ) | 1;
}

unsigned xorshift64::operator()()
{
	unsigned t = x ^ ( x << 13 );
	x = y;
	return y = ( y ^ ( y >> 5 ) ) ^ ( t ^ ( t >> 17 ) );
}

unsigned xorshift64::min() { return 0; }
unsigned xorshift64::max() { return std::numeric_limits<unsigned>::max(); }
} // namespace spice::util
