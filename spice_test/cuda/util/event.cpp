#include <gtest/gtest.h>

#include <spice/cuda/util/event.h>
#include <spice/util/stdint.h>


using namespace spice::cuda::util;


void dummy_kernel( cudaStream_t s );

template <typename T>
struct Event : ::testing::Test
{
};
using events = ::testing::Types<sync_event, time_event>;
TYPED_TEST_CASE( Event, events );

TYPED_TEST( Event, Ctor )
{
	{
		TypeParam e;
		ASSERT_TRUE( e.query() == cudaSuccess );

		for( int_ i = 0; i < 10; i++ )
		{
			dummy_kernel( 0 );
			e.record();
			EXPECT_TRUE( e.query() == cudaErrorNotReady ) << "Test is timing-dependent, repeat it";
		}

		e.synchronize();
		ASSERT_TRUE( e.query() == cudaSuccess );
	}
}

TEST( EventTime, ElapsedTime )
{
	time_event start, stop;

	for( size_ i = 0; i < 100; i++ )
	{
		start.record();
		dummy_kernel( 0 );
		stop.record();
		stop.synchronize();

		ASSERT_GT( stop.elapsed_time( start ), 0.0 );
		ASSERT_LT( start.elapsed_time( stop ), 0.0 );
	}
}