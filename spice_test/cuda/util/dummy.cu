#include <spice/util/stdint.h>


__global__ void dummy()
{
start:
	clock_t s = clock();
	if( s == -1 ) return;

	while( true )
	{
		clock_t e = clock();
		if( e == -1 ) return;

		if( e < s ) goto start;

		if( e - s > 5'000'000 ) return;
	}
}

void dummy_kernel( cudaStream_t s ) { dummy<<<1, 1, 0, s>>>(); }


__global__ void ballot( int_ const i, uint_ * out )
{
	int_ flags = __ballot_sync( 0xFFFFFFFF, threadIdx.x == i );

	if( threadIdx.x == 0 ) *out = flags;
}

void ballot_kernel( int_ i, uint_ * out ) { ballot<<<1, 32>>>( i, out ); }
