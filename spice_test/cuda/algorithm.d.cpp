#include <gtest/gtest.h>

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/dev_ptr.h>
#include <spice/util/adj_list.h>


using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


TEST( dAlgorithm, AdjList )
{
	auto const desc = neuron_group(
	    {10, 20, 30}, {{0, 0, 0.5f}, {0, 1, 0.1f}, {0, 2, 0.5f}, {1, 0, 1.0f}, {1, 2, 0.5f}} );
	// A(10)
	// B(20)
	// C(30)
	// A->A = 50%
	// A->B = 10%
	// A->C = 50%
	// B->A = 100%
	// B->C = 50%
	std::vector<adj_list::int4> layout( 3 );
	auto const deg = adj_list::generate( desc, layout );

	dev_ptr<int> d_e( deg * 60 );
	dev_ptr<adj_list::int4> d_layout( layout );
	generate_rnd_adj_list(
	    d_layout.data(),
	    narrow_int<int>( d_layout.size() ),
	    60,
	    narrow_int<int>( deg ),
	    d_e.data() );
	cudaDeviceSynchronize();

	adj_list adj( 60, deg, d_e.data() );

	for( std::size_t i = 0; i < 10; i++ )
	{
		int prev = -1;
		for( auto n : adj.neighbors( i ) )
		{
			ASSERT_TRUE( n >= 0 && n < 60 );
			ASSERT_GT( n, prev );
			prev = n;
		}
	}

	for( std::size_t i = 10; i < 30; i++ )
	{
		int prev = -1;
		for( auto n : adj.neighbors( i ) )
		{
			ASSERT_TRUE( n < 10 || n >= 30 );
			ASSERT_GT( n, prev );
			prev = n;
		}
	}

	for( std::size_t i = 30; i < 60; i++ )
		ASSERT_EQ( adj.neighbors( i ).size(), 0 );
}


void ballot_kernel( int i, unsigned * out );

TEST( dAlgorithm, bitfield )
{
	dev_ptr<unsigned> out( 1 );
	ballot_kernel( 7, out.data() );
	cudaDeviceSynchronize();

	ASSERT_EQ( out[0], 128u );
}