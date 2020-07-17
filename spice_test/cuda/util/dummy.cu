__global__ void dummy()
{
	clock_t s = clock();

	while( clock() - s < 1'000'000 ) {} // ~1ms
}

void dummy_kernel( cudaStream_t s ) { dummy<<<1, 1, 0, s>>>(); }


__global__ void ballot( int const i, unsigned * out )
{
	int flags = __ballot_sync( 0xFFFFFFFF, threadIdx.x == i );

	if( threadIdx.x == 0 ) *out = flags;
}

void ballot_kernel( int i, unsigned * out ) { ballot<<<1, 32>>>( i, out ); }