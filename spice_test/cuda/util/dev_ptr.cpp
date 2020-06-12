#include <gtest/gtest.h>

#include <spice/cuda/util/dev_ptr.h>

#include <spice/cuda/util/dev_var.h>
#include <spice/cuda/util/stream.h>


using namespace spice::cuda::util;


void dummy_kernel( cudaStream_t s );


TEST( DevPtr, Ctor )
{
	dev_ptr<int> y( 1 );
	ASSERT_EQ( y.size(), 1 );
	ASSERT_EQ( y.capacity(), 1 );
#pragma warning( push )
#pragma warning( disable : 4389 ) // "signed/unsigned mismatch" size() >= 0
	ASSERT_EQ( y.size_in_bytes(), sizeof( int ) );
#pragma warning( pop )
	ASSERT_NE( y.data(), nullptr );
	ASSERT_NE( y.begin(), nullptr );
	ASSERT_NE( y.cbegin(), nullptr );
	ASSERT_NE( y.end(), nullptr );
	ASSERT_NE( y.cend(), nullptr );
	ASSERT_EQ( std::distance( y.begin(), y.end() ), 1 );
	ASSERT_EQ( std::distance( y.cbegin(), y.cend() ), 1 );
	ASSERT_EQ( static_cast<nonstd::span<int const>>( y ).size(), 1 );

	y[0] = 5;
	ASSERT_EQ( y[0], 5 );
	for( auto i : y ) ASSERT_EQ( i, 5 );
	for( auto i : static_cast<nonstd::span<int const>>( y ) ) ASSERT_EQ( i, 5 );
}

TEST( DevPtr, Copy )
{
	std::vector<int> tmp{1, 2, 3, 4, 5};
	std::vector<int> tmp2{0, 1, 2, 3, 4};
	dev_ptr<int> x( tmp );
	dev_ptr<int> y( x );

	ASSERT_EQ( x, x );
	ASSERT_EQ( x, y );
	ASSERT_EQ( y, x );
	ASSERT_EQ( x, tmp );
	ASSERT_EQ( tmp, x );

	ASSERT_NE( x, tmp2 );
	ASSERT_NE( tmp2, x );
}

TEST( DevPtr, Resize )
{
	dev_ptr<int> x;
	ASSERT_EQ( x.size(), 0 );
	ASSERT_EQ( x.capacity(), 0 );

	x.resize( 123 );
	ASSERT_EQ( x.size(), 123 );
	ASSERT_EQ( x.capacity(), 123 );

	x.resize( 1 );
	ASSERT_EQ( x.size(), 1 );
	ASSERT_EQ( x.capacity(), 123 );
}

TEST( DevPtr, Zero )
{
	dev_ptr<float> x( 1 );
	x[0] = 5;
	x.zero();
	ASSERT_EQ( x.size(), 1 );
	ASSERT_EQ( x[0], 0 );

	x[0] = 7;
	x.zero_async();
	cudaDeviceSynchronize();
	ASSERT_EQ( x.size(), 1 );
	ASSERT_EQ( x[0], 0 );
}

TEST( DevPtr, Attach )
{
	stream sx, sg;
	dev_ptr<int> x( 1 );

	x.attach( sx ); // *
	sx.synchronize();

	dummy_kernel( sg );
	x[0] = 5; // causes SEH without (*)
}