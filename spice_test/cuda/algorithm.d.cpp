#include <gtest/gtest.h>

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/dbuffer.h>
#include <spice/util/adj_list.h>


using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


TEST( dAlgorithm, AdjList )
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
	dbuffer<int> d_e( 60 * desc.max_degree() );
	auto deg = desc.max_degree();
	generate_rnd_adj_list( desc, d_e.data() );
	cudaDeviceSynchronize();

	std::vector<int> h_e( d_e );
	adj_list adj( 60, deg, h_e.data() );

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

	for( std::size_t i = 30; i < 60; i++ ) ASSERT_EQ( adj.neighbors( i ).size(), 0u );
}