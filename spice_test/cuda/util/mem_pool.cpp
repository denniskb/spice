#include <gtest/gtest.h>

#include <spice/cuda/util/mem_pool.h>

#include <spice/util/span.hpp>


using namespace spice::cuda::util;


TEST( MemPool, Ctor )
{
	{
		ASSERT_THROW( mem_pool x( 0 ), std::bad_alloc );
	}

	{
		mem_pool x( 10 );
		ASSERT_EQ( x.size(), 0 );
		ASSERT_EQ( x.capacity(), 10 );
	}
}

TEST( MemPool, Alloc )
{
	mem_pool x( 10 );

	x.alloc( 5 );
	ASSERT_EQ( x.size(), 5 );
	ASSERT_EQ( x.capacity(), 10 );

	x.alloc( 5 );
	ASSERT_EQ( x.size(), 10 );
	ASSERT_EQ( x.capacity(), 10 );
}