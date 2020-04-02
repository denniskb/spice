#pragma once


namespace spice
{
namespace cuda
{
namespace util
{
__device__ inline unsigned hash( unsigned x )
{
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( ( x >> 16 ) ^ x ) * 0x45d9f3b;
	x = ( x >> 16 ) ^ x;

	return x;
}

class xorwow
{
public:
	__device__ xorwow( unsigned seed = 1337 )
	    : x( hash( seed ) )
	{
	}

	// @return iid in [0, 1]
	__device__ float operator()()
	{
		unsigned t = ( x ^ ( x >> 2 ) );

		x = y;
		y = z;
		z = w;
		w = v;
		v = ( v ^ ( v << 4 ) ) ^ ( t ^ ( t << 1 ) );
		d += 362437u;

		return ( d + v ) * ( 1.0f / 4294967295.0f );
	}

private:
	unsigned x;
	unsigned y = 362436069u;
	unsigned z = 521288629u;
	unsigned w = 88675123u;
	unsigned v = 5783321u;
	unsigned d = 6615241u;
};
} // namespace util
} // namespace cuda
} // namespace spice
