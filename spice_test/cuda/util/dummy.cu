#include <spice/util/stdint.h>


__global__ void dummy()
{
	clock_t s = clock();

	while( clock() - s < 1'000'000 ) {} // ~1ms
}

void dummy_kernel( cudaStream_t s ) { dummy<<<1, 1, 0, s>>>(); }


__global__ void ballot( int_ const i, uint_ * out )
{
	int_ flags = __ballot_sync( 0xFFFFFFFF, threadIdx.x == i );

	if( threadIdx.x == 0 ) *out = flags;
}

void ballot_kernel( int_ i, uint_ * out ) { ballot<<<1, 32>>>( i, out ); }
