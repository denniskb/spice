#include <gtest/gtest.h>

#include <spice/util/adj_list.h>


using namespace spice::util;


TEST( AdjList, Ctor )
{
	{
		adj_list x;
		ASSERT_EQ( x.num_nodes(), 0u );
		ASSERT_EQ( x.max_degree(), 0u );
		ASSERT_EQ( x.num_edges(), 0u );

		ASSERT_EQ( x.edges(), nullptr );
	}

	{
		std::vector<int> e;
		layout desc( 100, 0 );
		auto const deg = desc.max_degree();
		adj_list::generate( desc, e );

		adj_list x( 100, deg, e.data() );

		ASSERT_EQ( x.num_nodes(), 100u );
		ASSERT_EQ( x.max_degree(), 0u );
		ASSERT_EQ( x.num_edges(), 0u );

		for( int i = 0; i < 100; i++ ) ASSERT_EQ( x.neighbors( i ).size(), 0u );
	}

	{
		std::vector<int> e;
		layout desc( 100, 1 );
		auto const deg = desc.max_degree();
		adj_list::generate( desc, e );

		adj_list x( 100, deg, e.data() );

		ASSERT_EQ( x.num_nodes(), 100u );
		ASSERT_EQ( x.max_degree(), 128u );
		ASSERT_EQ( x.num_edges(), deg * 100 );

		for( int i = 0; i < 100; i++ )
		{
			ASSERT_EQ( x.neighbors( i ).size(), 100u );

			int j = 0;
			for( auto n : x.neighbors( i ) )
			{
				ASSERT_EQ( n, j );
				ASSERT_EQ( x.edge_index( i, j ), deg * i + j );
				++j;
			}
		}
	}

	{
		std::vector<int> e;
		layout desc( 100, 0.5f );
		auto const deg = desc.max_degree();
		adj_list::generate( desc, e );

		adj_list x( 100, deg, e.data() );

		ASSERT_EQ( x.num_nodes(), 100u );
		ASSERT_LE( x.max_degree(), 128u );
		ASSERT_EQ( x.num_edges(), deg * 100 );

		for( int i = 0; i < 100; i++ )
		{
			int prev = -1;
			for( auto y : x.neighbors( i ) )
			{
				ASSERT_GE( y, 0 );
				ASSERT_LT( y, 100 );

				ASSERT_GT( y, prev );
				prev = y;
			}
		}
	}

	{
		auto desc = layout(
		    { 10, 20, 30 },
		    { { 0, 0, 0.5f }, { 0, 1, 0.1f }, { 0, 2, 0.5f }, { 1, 0, 1.0f }, { 1, 2, 0.5f } } );
		// A(10)
		// B(20)
		// C(30)
		// A->A = 50%
		// A->B = 10%
		// A->C = 50%
		// B->A = 100%
		// B->C = 50%
		std::vector<int> e;
		adj_list::generate( desc, e );
		adj_list adj( 60, desc.max_degree(), e.data() );
		auto const deg = desc.max_degree();

		for( std::size_t i = 0; i < 10; i++ )
		{
			auto const degree = adj.neighbors( i ).size();
			ASSERT_TRUE( degree >= 0 && degree <= 60 );
			EXPECT_TRUE( degree >= 12 && degree <= 32 ) << degree << " (rng)";

			int prev = -1;
			int j = 0;
			for( auto n : adj.neighbors( i ) )
			{
				ASSERT_TRUE( n >= 0 && n < 60 );
				ASSERT_GT( n, prev );
				prev = n;

				ASSERT_EQ( adj.edge_index( i, j ), i * deg + j );
				j++;
			}
		}

		for( std::size_t i = 10; i < 30; i++ )
		{
			auto const degree = adj.neighbors( i ).size();
			ASSERT_TRUE( degree >= 10 && degree <= 40 );
			EXPECT_TRUE( degree >= 15 && degree <= 35 ) << degree << " (rng)";

			int prev = -1;
			int j = 0;
			for( auto n : adj.neighbors( i ) )
			{
				ASSERT_TRUE( n < 10 || n >= 30 );
				ASSERT_GT( n, prev );
				prev = n;

				ASSERT_EQ( adj.edge_index( i, j ), i * deg + j );
				j++;
			}
		}

		for( std::size_t i = 30; i < 60; i++ )
		{
			auto const degree = adj.neighbors( i ).size();
			ASSERT_EQ( degree, 0u );

			ASSERT_EQ( adj.neighbors( i ).size(), 0u );
		}
	}
}