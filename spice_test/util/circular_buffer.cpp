#include <gtest/gtest.h>

#include <spice/util/circular_buffer.h>


using namespace spice::util;


TEST( CircBuffer, Ctor )
{
	{
		circular_buffer<int> x( 0 );
		ASSERT_EQ( x.size(), 0 );
		ASSERT_EQ( x.end() - x.begin(), 0 );
	}

	{
		circular_buffer<int> x( 10 );
		ASSERT_EQ( x.size(), 10 );
		ASSERT_EQ( x.end() - x.begin(), 10 );

		for( int i = 0; i < 10; i++ ) ASSERT_EQ( x[i], 0 );

		for( auto j : x ) ASSERT_EQ( j, 0 );
	}
}

TEST( CircBuffer, Access )
{
	circular_buffer<int> x( 3 );

	x[0] = 1;
	x[1] = 2;
	x[2] = 3;
	ASSERT_EQ( x[0], 1 );
	ASSERT_EQ( x[1], 2 );
	ASSERT_EQ( x[2], 3 );

	x[4] = 4;
	ASSERT_EQ( x[4], 4 );
	ASSERT_EQ( x[1], 4 );
	ASSERT_EQ( x[-2], 4 );
	ASSERT_EQ( x[-3], 1 );
	ASSERT_EQ( x[124], 4 );
	ASSERT_EQ( x[0], 1 );
	ASSERT_EQ( x[2], 3 );
}