#include <gtest/gtest.h>

#include <spice/cuda/util/stream.h>

#include <spice/cuda/util/dev_var.h>
#include <spice/cuda/util/event.h>
#include <spice/cuda/util/memory.h>

#include <functional>
#include <memory>


using namespace spice::util;
using namespace spice::cuda::util;


// Non-zero cost work on specified stream
void dummy_kernel( cudaStream_t s );
void dummy_copy( cudaStream_t s )
{
	int const N = 3'000'000;
	static std::unique_ptr<int, cudaError_t ( * )( void * )> host(
	    static_cast<int *>( cuda_malloc_host( N * 4 ) ), cudaFree );
	static std::unique_ptr<int, cudaError_t ( * )( void * )> device(
	    static_cast<int *>( cuda_malloc( N * 4 ) ), cudaFree );

	cudaMemcpyAsync( device.get(), host.get(), N * 4, cudaMemcpyDefault, s ); // ~1ms
}

TEST( Stream, Ctor )
{
	stream s1, s2;
	event s1start, s1stop, s2start, s2stop;

	s1start.record( s1 );
	dummy_kernel( s1 );
	s1stop.record( s1 );

	s2start.record( s2 );
	dummy_copy( s2 );
	s2stop.record( s2 );

	s1stop.synchronize();
	s2stop.synchronize();

	ASSERT_TRUE( s1stop.elapsed_time( s1start ) > 0.0f );
	ASSERT_TRUE( s2stop.elapsed_time( s2start ) > 0.0f );
}

TEST( Stream, Query )
{
	stream s;

	for( std::size_t i = 0; i < 100; i++ )
	{
		dummy_kernel( s );
		EXPECT_EQ( s.query(), cudaErrorNotReady );

		s.synchronize();
		ASSERT_EQ( s.query(), cudaSuccess );
	}
}

TEST( Stream, Wait )
{
	stream s1, s2;
	event s1start, s1stop, s2start, s2stop;

	for( std::size_t i = 0; i < 100; i++ )
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

		ASSERT_GT( s1stop.elapsed_time( s1start ), 0.0f );
		ASSERT_GT( s2stop.elapsed_time( s2start ), 0.0f );

		ASSERT_GE( s2start.elapsed_time( s1stop ), 0.0f );
	}

	for( std::size_t i = 0; i < 100; i++ )
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

		ASSERT_GT( s1stop.elapsed_time( s1start ), 0.0f );
		ASSERT_GT( s2stop.elapsed_time( s2start ), 0.0f );

		EXPECT_LT( s2start.elapsed_time( s1stop ), 0.0f ) << "test is timing-dependent, repeat it";
	}
}

TEST( Stream, Synchronize )
{
	stream s1, s2;
	event s1start, s1stop, s2start, s2stop;

	for( std::size_t i = 0; i < 100; i++ )
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

		ASSERT_GE( s2start.elapsed_time( s1stop ), 0.0f );
	}

	for( std::size_t i = 0; i < 100; i++ )
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
		EXPECT_LT( s2start.elapsed_time( s1stop ), 0.0f ) << "test is time-dependent, repeat it";
	}
}

TEST( Stream, Default )
{
	ASSERT_EQ( static_cast<cudaStream_t>( stream::default_stream() ), nullptr );
}