#include <gtest/gtest.h>

#include <spice/util/layout.h>


using namespace spice::util;


TEST( Layout, DefaultCtor )
{
	layout l( 1, 0.0f );

	ASSERT_EQ( l.size(), 1 );
	ASSERT_EQ( l.connections().size(), 0 );
	ASSERT_EQ( l.max_degree(), 0 );
}

TEST( Layout, DefaultCtor2 )
{
	std::vector<std::tuple<std::size_t, std::size_t, float>> connects;
	layout l( { 1 }, connects );

	ASSERT_EQ( l.size(), 1 );
	ASSERT_EQ( l.connections().size(), 0 );
	ASSERT_EQ( l.max_degree(), 0 );
}

TEST( Layout, FromNP )
{
	layout l( 100, 0.25f );

	ASSERT_EQ( l.size(), 100 );
	ASSERT_EQ( l.connections().size(), 1 );
	ASSERT_EQ( l.connections()[0], std::make_tuple( 0, 100, 0, 100, 0.25f ) );
}

TEST( Layout, FromPopConnect )
{
	layout l( { 10, 20, 30 }, { { 1, 0, 0.5f }, { 2, 1, 0.25f }, { 1, 2, 0.125f } } );

	ASSERT_EQ( l.size(), 60 );
	ASSERT_EQ( l.connections().size(), 3 );
	ASSERT_EQ( l.connections()[0], std::make_tuple( 10, 30, 0, 10, 0.5f ) );
	ASSERT_EQ( l.connections()[1], std::make_tuple( 10, 30, 30, 60, 0.125f ) );
	ASSERT_EQ( l.connections()[2], std::make_tuple( 30, 60, 10, 30, 0.25f ) );
}