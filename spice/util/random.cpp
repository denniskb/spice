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
xorwow::xorwow( unsigned seed /* = 1337 */ )
    : x( hash( seed ) )
{
}

unsigned xorwow::operator()()
{
	unsigned const t = ( x ^ ( x >> 2 ) );

	d += 362437u;
	x = y;
	y = z;
	z = w;
	w = v;
	v = ( v ^ ( v << 4 ) ) ^ ( t ^ ( t << 1 ) );

	return d + v;
}

unsigned xorwow::min() { return 0; }
unsigned xorwow::max() { return std::numeric_limits<unsigned>::max(); }
} // namespace spice::util
