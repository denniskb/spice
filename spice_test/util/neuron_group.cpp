#include <gtest/gtest.h>

#include <spice/util/neuron_group.h>


using namespace spice::util;


TEST( NeuronGroup, Ctor )
{
	{
		std::vector<neuron_group::edge> const C{
		    {0, 1, 0.1f}, {0, 2, 0.2f}, {1, 1, 0.3f}, {1, 2, 0.4f}, {2, 1, 0.5f}, {2, 2, 0.6f}};

		neuron_group x( {5, 2, 3}, C );

		ASSERT_EQ( 3, x.num_groups() );

		ASSERT_EQ( 10, x.size() );
		ASSERT_EQ( 5, x.size( 0 ) );
		ASSERT_EQ( 2, x.size( 1 ) );
		ASSERT_EQ( 3, x.size( 2 ) );

		ASSERT_EQ( 0, x.first( 0 ) );
		ASSERT_EQ( 5, x.first( 1 ) );
		ASSERT_EQ( 7, x.first( 2 ) );

		ASSERT_EQ( 5, x.last( 0 ) );
		ASSERT_EQ( 7, x.last( 1 ) );
		ASSERT_EQ( 10, x.last( 2 ) );

		{
			int i = 0;
			for( auto const & c : x.connections() ) ASSERT_EQ( c, C[i++] );
		}

		ASSERT_EQ( 2, x.neighbors( 0 ).size() );
		ASSERT_EQ( 0, std::get<0>( x.neighbors( 0 )[0] ) );
		ASSERT_EQ( 1, std::get<1>( x.neighbors( 0 )[0] ) );
		ASSERT_EQ( 0.1f, std::get<2>( x.neighbors( 0 )[0] ) );
		ASSERT_EQ( 0, std::get<0>( x.neighbors( 0 )[1] ) );
		ASSERT_EQ( 2, std::get<1>( x.neighbors( 0 )[1] ) );
		ASSERT_EQ( 0.2f, std::get<2>( x.neighbors( 0 )[1] ) );

		ASSERT_EQ( 2, x.neighbors( 1 ).size() );
		ASSERT_EQ( 1, std::get<0>( x.neighbors( 1 )[0] ) );
		ASSERT_EQ( 1, std::get<1>( x.neighbors( 1 )[0] ) );
		ASSERT_EQ( 0.3f, std::get<2>( x.neighbors( 1 )[0] ) );
		ASSERT_EQ( 1, std::get<0>( x.neighbors( 1 )[1] ) );
		ASSERT_EQ( 2, std::get<1>( x.neighbors( 1 )[1] ) );
		ASSERT_EQ( 0.4f, std::get<2>( x.neighbors( 1 )[1] ) );

		ASSERT_EQ( 2, x.neighbors( 2 ).size() );
		ASSERT_EQ( 2, std::get<0>( x.neighbors( 2 )[0] ) );
		ASSERT_EQ( 1, std::get<1>( x.neighbors( 2 )[0] ) );
		ASSERT_EQ( 0.5f, std::get<2>( x.neighbors( 2 )[0] ) );
		ASSERT_EQ( 2, std::get<0>( x.neighbors( 2 )[1] ) );
		ASSERT_EQ( 2, std::get<1>( x.neighbors( 2 )[1] ) );
		ASSERT_EQ( 0.6f, std::get<2>( x.neighbors( 2 )[1] ) );
	}

	{
		neuron_group x( 100, 0.1f );

		ASSERT_EQ( 1, x.num_groups() );

		ASSERT_EQ( 100, x.size() );
		ASSERT_EQ( 100, x.size( 0 ) );

		ASSERT_EQ( 0, x.first( 0 ) );
		ASSERT_EQ( 100, x.last( 0 ) );

		for( auto const & c : x.connections() )
			ASSERT_TRUE(
			    std::get<0>( c ) == 0 && std::get<1>( c ) == 0 && std::get<2>( c ) == 0.1f );

		ASSERT_EQ( 1, x.neighbors( 0 ).size() );
		ASSERT_EQ( 0, std::get<0>( x.neighbors( 0 )[0] ) );
		ASSERT_EQ( 0, std::get<1>( x.neighbors( 0 )[0] ) );
		ASSERT_EQ( 0.1f, std::get<2>( x.neighbors( 0 )[0] ) );
	}
}