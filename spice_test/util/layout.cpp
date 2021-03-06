#include <gtest/gtest.h>

#include <spice/util/layout.h>


using namespace spice::util;


TEST( Layout, DefaultCtor )
{
	layout l( 1, 0.0f );

	ASSERT_EQ( l.size(), 1u );
	ASSERT_EQ( l.max_degree(), 0u );
	ASSERT_EQ( l.connections().size(), 1u );
	ASSERT_EQ( l.connections()[0], std::make_tuple( 0, 1, 0, 1, 0.0f ) );
}

TEST( Layout, DefaultCtor2 )
{
	std::vector<std::tuple<size_, size_, float>> connects;
	layout l( { 1 }, connects );

	ASSERT_EQ( l.size(), 1u );
	ASSERT_EQ( l.connections().size(), 0u );
	ASSERT_EQ( l.max_degree(), 0u );
}

TEST( Layout, FromNP )
{
	layout l( 100, 0.25f );

	ASSERT_EQ( l.size(), 100u );
	ASSERT_EQ( l.connections().size(), 1u );
	ASSERT_EQ( l.connections()[0], std::make_tuple( 0, 100, 0, 100, 0.25f ) );
}

TEST( Layout, FromPopConnect )
{
	layout l( { 10, 20, 30 }, { { 1, 0, 0.5f }, { 2, 1, 0.25f }, { 1, 2, 0.125f } } );

	ASSERT_EQ( l.size(), 60u );
	ASSERT_EQ( l.connections().size(), 3u );
	ASSERT_EQ( l.connections()[0], std::make_tuple( 10, 30, 0, 10, 0.5f ) );
	ASSERT_EQ( l.connections()[1], std::make_tuple( 10, 30, 30, 60, 0.125f ) );
	ASSERT_EQ( l.connections()[2], std::make_tuple( 30, 60, 10, 30, 0.25f ) );
}

TEST( Layout, Slice )
{
	{
		layout l( 10, 0.5f );
		{
			auto s = l.cut( l.static_load_balance( 1, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 10 );
			ASSERT_EQ( s.part.connections(), l.connections() );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 5 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 0, 5, 0.5f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 1 ) );
			ASSERT_EQ( s.first, 5 );
			ASSERT_EQ( s.last, 10 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 5, 10, 0.5f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 3, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 3 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 0, 3, 0.5f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 3, 1 ) );
			ASSERT_EQ( s.first, 3 );
			ASSERT_EQ( s.last, 6 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 3, 6, 0.5f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 3, 2 ) );
			ASSERT_EQ( s.first, 6 );
			ASSERT_EQ( s.last, 10 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 6, 10, 0.5f ) );
		}
	}

	{
		layout l( { 10, 10 }, { { 0, 1, 0.25f }, { 1, 1, 0.75f } } );
		{
			auto s = l.cut( l.static_load_balance( 1, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 20 );
			ASSERT_EQ( s.part.connections(), l.connections() );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 14 );
			ASSERT_EQ( s.part.connections().size(), 2u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 10, 14, 0.25f ) );
			ASSERT_EQ( s.part.connections()[1], std::make_tuple( 10, 20, 10, 14, 0.75f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 1 ) );
			ASSERT_EQ( s.first, 14 );
			ASSERT_EQ( s.last, 20 );
			ASSERT_EQ( s.part.connections().size(), 2u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 14, 20, 0.25f ) );
			ASSERT_EQ( s.part.connections()[1], std::make_tuple( 10, 20, 14, 20, 0.75f ) );
		}
	}

	{
		layout l( { 10, 10, 10 }, { { 0, 0, 0.5f }, { 2, 2, 0.25f }, { 1, 2, 0.25f } } );
		{
			auto s = l.cut( l.static_load_balance( 1, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 30 );
			ASSERT_EQ( s.part.connections(), l.connections() );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 15 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 0, 10, 0.5f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 1 ) );
			ASSERT_EQ( s.first, 15 );
			ASSERT_EQ( s.last, 30 );
			ASSERT_EQ( s.part.connections().size(), 2u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 10, 20, 20, 30, 0.25f ) );
			ASSERT_EQ( s.part.connections()[1], std::make_tuple( 20, 30, 20, 30, 0.25f ) );
		}
	}

	{
		layout l( { 10, 10 }, { { 0, 0, 0.1f }, { 1, 1, 0.9f } } );
		{
			auto s = l.cut( l.static_load_balance( 1, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 20 );
			ASSERT_EQ( s.part.connections(), l.connections() );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 0 ) );
			ASSERT_EQ( s.first, 0 );
			ASSERT_EQ( s.last, 14 );
			ASSERT_EQ( s.part.connections().size(), 2u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 10, 0, 10, 0.1f ) );
			ASSERT_EQ( s.part.connections()[1], std::make_tuple( 10, 20, 10, 14, 0.9f ) );
		}
		{
			auto s = l.cut( l.static_load_balance( 2, 1 ) );
			ASSERT_EQ( s.first, 14 );
			ASSERT_EQ( s.last, 20 );
			ASSERT_EQ( s.part.connections().size(), 1u );
			ASSERT_EQ( s.part.connections()[0], std::make_tuple( 10, 20, 14, 20, 0.9f ) );
		}
	}

	{
		layout l( 66'000, 1.0f );
		auto s = l.cut( l.static_load_balance( 2, 0 ) );
		ASSERT_EQ( s.first, 0 );
		ASSERT_EQ( s.last, 33'000 );
		ASSERT_EQ( s.part.connections().size(), 1 );
		ASSERT_EQ( s.part.connections()[0], std::make_tuple( 0, 66'000, 0, 33'000, 1.0f ) );
	}
}

TEST( Layout, SliceStrided )
{
	{
		layout const l( 100, 0.123f );
		{
			auto s = l.cut( 100, 1, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 100, 0.123f ) );
		}

		{
			auto s = l.cut( 50, 1, 0 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 50, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 50, 100, 0.123f ) );
		}

		{
			auto s = l.cut( 34, 1, 0 );
			ASSERT_EQ( s.connections().size(), 3 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 34, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 34, 68, 0.123f ) );
			ASSERT_EQ( s.connections()[2], std::make_tuple( 0, 100, 68, 100, 0.123f ) );
		}

		{
			auto s = l.cut( 50, 2, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 50, 0.123f ) );
		}
		{
			auto s = l.cut( 50, 2, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 50, 100, 0.123f ) );
		}

		{
			auto s = l.cut( 51, 2, 0 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 51, 0.123f ) );
		}
		{
			auto s = l.cut( 51, 2, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 51, 100, 0.123f ) );
		}

		{
			auto s = l.cut( 49, 2, 0 );
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 49, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 98, 100, 0.123f ) );
		}
		{
			auto s = l.cut( 49, 2, 1 );
			ASSERT_EQ( s.connections().size(), 1 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 49, 98, 0.123f ) );
		}

		{
			auto s = l.cut(16, 3, 0);
			ASSERT_EQ( s.connections().size(), 3 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 0, 16, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 48, 64, 0.123f ) );
			ASSERT_EQ( s.connections()[2], std::make_tuple( 0, 100, 96, 100, 0.123f ) );
		}
		{
			auto s = l.cut(16, 3, 1);
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 16, 32, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 64, 80, 0.123f ) );
		}
		{
			auto s = l.cut(16, 3, 2);
			ASSERT_EQ( s.connections().size(), 2 );
			ASSERT_EQ( s.connections()[0], std::make_tuple( 0, 100, 32, 48, 0.123f ) );
			ASSERT_EQ( s.connections()[1], std::make_tuple( 0, 100, 80, 96, 0.123f ) );
		}
	}
}