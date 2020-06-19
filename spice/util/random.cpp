#include "random.h"


static unsigned hash( unsigned x )
{
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( x >> 16 ) ^ x;

	return x;
}


namespace spice::util
{
xorshift::xorshift( unsigned seed /* = 1337 */ )
    : a( hash( seed + 1 ) | 1 )
{
}

unsigned xorshift::operator()()
{
	unsigned x = a;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return ( a = x );
}

unsigned xorshift::min() { return 0; }
unsigned xorshift::max() { return std::numeric_limits<unsigned>::max(); }


xorwow::xorwow( unsigned seed /* = 1337 */ )
    : a( hash( seed + 1 ) | 1 )
    , b( hash( hash( seed + 1 ) ) | 1 )
    , c( hash( hash( hash( seed + 1 ) ) ) | 1 )
    , d( hash( hash( hash( hash( seed + 1 ) ) ) ) | 1 )
{
}

unsigned xorwow::operator()()
{
	unsigned t = d;
	unsigned s = a;

	d = c;
	c = b;
	b = s;

	t ^= t >> 2;
	t ^= t << 1;
	t ^= s ^ ( s << 4 );
	a = t;

	counter += 362437;

	return t + counter;
}

unsigned xorwow::min() { return 0; }
unsigned xorwow::max() { return std::numeric_limits<unsigned>::max(); }
} // namespace spice::util
