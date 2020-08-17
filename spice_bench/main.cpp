#include <benchmark/benchmark.h>

#include <cuda_runtime.h>


int main( int argc, char ** argv )
{
	::benchmark::Initialize( &argc, argv );
	// cudaSetDevice( 1 );
	return ::benchmark::RunSpecifiedBenchmarks() > 0;
}