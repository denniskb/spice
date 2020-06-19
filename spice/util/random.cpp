#include "random.h"


#define rotl( x, k ) ( ( x << k ) | ( x >> ( 32 - k ) ) )

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


xoroshiro64::xoroshiro64( unsigned long long seed )
{
	auto h = hash( seed + 1 );
	x = (unsigned)h | 1;
	y = (unsigned)( h >> 32 ) | 1;
}

unsigned xoroshiro64::operator()()
{
	unsigned result = x * 0x9E3779BB;

	y ^= x;
	x = rotl( x, 26 ) ^ y ^ ( y << 9 );
	y = rotl( y, 13 );

	return result;
}

unsigned xoroshiro64::min() { return 0; }
unsigned xoroshiro64::max() { return std::numeric_limits<unsigned>::max(); }
} // namespace spice::util
