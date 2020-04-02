#include <gtest/gtest.h>

#include <spice/cuda/util/event.h>


using namespace spice::cuda::util;


void dummy_kernel( cudaStream_t s );

TEST( Event, Ctor )
{
	event e;

	e.record();
	ASSERT_TRUE( e.query() == cudaErrorNotReady ); // no prior work submitted

	e.synchronize();
	ASSERT_TRUE( e.query() == cudaSuccess );
}

TEST( Event, ElapsedTime )
{
	event start, stop;

	for( std::size_t i = 0; i < 100; i++ )
	{
		start.record();
		dummy_kernel( 0 );
		stop.record();
		stop.synchronize();

		ASSERT_GT( stop.elapsed_time( start ), 0.0f );
		ASSERT_LT( start.elapsed_time( stop ), 0.0f );
	}
}