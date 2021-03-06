#include <benchmark/benchmark.h>

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/error.h>


using namespace spice::cuda::util;


int main( int argc, char ** argv )
{
	::benchmark::Initialize( &argc, argv );
	// cudaSetDevice( 1 );
	return ::benchmark::RunSpecifiedBenchmarks() == 0;
}