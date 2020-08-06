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

TEST( Layout, Slice )
{
	{
		layout l( 10, 0.5f );
		{
			layout s = l.slice( 1, 0 );
			ASSERT_EQ( s.connections(), l.connections() );
		}
		{
			layout s = l.slice( 2, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 0, 5, 0.5f ) );
		}
		{
			layout s = l.slice( 2, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 5, 10, 0.5f ) );
		}
		{
			layout s = l.slice( 3, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 0, 3, 0.5f ) );
		}
		{
			layout s = l.slice( 3, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 3, 6, 0.5f ) );
		}
		{
			layout s = l.slice( 3, 2 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 6, 10, 0.5f ) );
		}
	}

	{
		layout l( { 10, 10 }, { { 0, 1, 0.25f }, { 1, 1, 0.75f } } );
		{
			layout s = l.slice( 1, 0 );
			ASSERT_EQ( s.connections(), l.connections() );
		}
		{
			layout s = l.slice( 2, 0 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 10, 15, 0.25f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 10, 20, 10, 15, 0.75f ) );
		}
		{
			layout s = l.slice( 2, 1 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 15, 20, 0.25f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 10, 20, 15, 20, 0.75f ) );
		}
	}

	{
		layout l( { 10, 10, 10 }, { { 0, 0, 0.5f }, { 2, 2, 0.25f }, { 1, 2, 0.25f } } );
		{
			layout s = l.slice( 1, 0 );
			ASSERT_EQ( s.connections(), l.connections() );
		}
		{
			layout s = l.slice( 2, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 0, 10, 0.5f ) );
		}
		{
			layout s = l.slice( 2, 1 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 10, 20, 20, 30, 0.25f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 20, 30, 20, 30, 0.25f ) );
		}
	}

	{
		layout l( { 10, 10 }, { { 0, 0, 0.1f }, { 1, 1, 0.9f } } );
		{
			layout s = l.slice( 1, 0 );
			ASSERT_EQ( s.connections(), l.connections() );
		}
		{
			layout s = l.slice( 2, 0 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 10, 0, 10, 0.1f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 10, 20, 10, 14, 0.9f ) );
		}
		{
			layout s = l.slice( 2, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 10, 20, 14, 20, 0.9f ) );
		}
	}
}