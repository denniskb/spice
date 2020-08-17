#include <gtest/gtest.h>

#include <spice/util/meta.h>

#include <vector>


using namespace spice::util;


using test_types = type_list<int, int>;
enum attr
{
	One,
	Two
};


TEST( Meta, ForEach )
{
	{
		std::tuple<int, int> x( 1, 2 );
		for_each( x, []( auto & elem ) { elem *= 2; } );
		ASSERT_EQ( std::get<0>( x ), 2 );
		ASSERT_EQ( std::get<1>( x ), 4 );
	}

	{
		std::tuple<int, int> x( 0, 1 );
		for_each_i( x, []( auto elem, auto i ) { ASSERT_EQ( elem, i ); } );
	}
}

TEST( Meta, Map )
{
	{
		std::tuple<int, int> x( 1, 3 );
		auto y = map( x, []( auto elem ) { return elem * 1.5f; } );
		ASSERT_EQ( std::get<0>( y ), 1.5f );
		ASSERT_EQ( std::get<1>( y ), 4.5f );
	}
}