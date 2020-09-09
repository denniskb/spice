#include <gtest/gtest.h>

#include <spice/cuda/util/stream.h>

#include <spice/cuda/util/dvar.h>
#include <spice/cuda/util/event.h>
#include <spice/cuda/util/memory.h>

#include <functional>
#include <memory>


using namespace spice::cuda::util;


// Non-zero cost work on specified stream
void dummy_kernel( cudaStream_t s );

class Stream : public ::testing::Test
{
private:
	int_ const N = 10'000'000;
	std::unique_ptr<int, cudaError_t ( * )( void * )> host{ nullptr, cudaFreeHost };
	std::unique_ptr<int, cudaError_t ( * )( void * )> device{ nullptr, cudaFree };

protected:
	void SetUp() override
	{
		host.reset( static_cast<int_ *>( cuda_malloc_host( N ) ) );
		device.reset( static_cast<int_ *>( cuda_malloc( N ) ) );
	}

	void dummy_copy( cudaStream_t s )
	{
		cudaMemcpyAsync( device.get(), host.get(), N, cudaMemcpyDefault, s );
	}
};

TEST_F( Stream, Ctor )
{
	stream s1, s2;
	time_event s1start, s1stop, s2start, s2stop;

	s1start.record( s1 );
	dummy_kernel( s1 );
	s1stop.record( s1 );

	s2start.record( s2 );
	dummy_copy( s2 );
	s2stop.record( s2 );

	s1stop.synchronize();
	s2stop.synchronize();

	ASSERT_TRUE( s1stop.elapsed_time( s1start ) > 0.0 );
	ASSERT_TRUE( s2stop.elapsed_time( s2start ) > 0.0 );
}

TEST_F( Stream, Query )
{
	stream s;

	for( size_ i = 0; i < 10; i++ )
	{
		dummy_kernel( s );
		EXPECT_EQ( s.query(), cudaErrorNotReady );

		s.synchronize();
		ASSERT_EQ( s.query(), cudaSuccess );
	}
}

TEST_F( Stream, Wait )
{
	stream s1, s2;
	time_event s1start, s1stop, s2start, s2stop;

	for( size_ i = 0; i < 10; i++ )
	{
		s1start.record( s1 );
		dummy_kernel( s1 );
		s1stop.record( s1 );

		s2.wait( s1stop );
		s2start.record( s2 );
		dummy_copy( s2 );
		s2stop.record( s2 );

		s2stop.synchronize();

		ASSERT_EQ( s1stop.query(), cudaSuccess );

		ASSERT_GT( s1stop.elapsed_time( s1start ), 0.0 );
		ASSERT_GT( s2stop.elapsed_time( s2start ), 0.0 );

		ASSERT_GE( s2start.elapsed_time( s1stop ), 0.0 );
	}

	for( size_ i = 0; i < 10; i++ )
	{
		s1start.record( s1 );
		dummy_kernel( s1 );
		s1stop.record( s1 );

		// s2.wait(s1stop);
		s2start.record( s2 );
		dummy_copy( s2 );
		s2stop.record( s2 );

		s2stop.synchronize();
		s1stop.synchronize(); // ASSERT_TRUE(s1stop.query() == cudaSuccess);

		ASSERT_GT( s1stop.elapsed_time( s1start ), 0.0 );
		ASSERT_GT( s2stop.elapsed_time( s2start ), 0.0 );

		EXPECT_LT( s2start.elapsed_time( s1stop ), 0.0 ) << "test is timing-dependent, repeat it";
	}
}

TEST_F( Stream, Synchronize )
{
	stream s1, s2;
	time_event s1start, s1stop, s2start, s2stop;

	for( size_ i = 0; i < 10; i++ )
	{
		s1start.record( s1 );
		dummy_kernel( s1 );
		s1stop.record( s1 );

		s1.synchronize();

		s2start.record( s2 );
		dummy_copy( s2 );
		s2stop.record( s2 );

		s2.synchronize();
		ASSERT_EQ( s1stop.query(), cudaSuccess );
		ASSERT_EQ( s2stop.query(), cudaSuccess );

		ASSERT_GE( s2start.elapsed_time( s1stop ), 0.0 );
	}

	for( size_ i = 0; i < 10; i++ )
	{
		s1start.record( s1 );
		dummy_kernel( s1 );
		s1stop.record( s1 );

		// s1.synchronize();

		s2start.record( s2 );
		dummy_copy( s2 );
		s2stop.record( s2 );

		s2.synchronize();
		s1.synchronize();
		EXPECT_LT( s2start.elapsed_time( s1stop ), 0.0 ) << "test is time-dependent, repeat it";
	}
}

TEST_F( Stream, Default )
{
	ASSERT_EQ( static_cast<cudaStream_t>( stream::default_stream() ), nullptr );
}