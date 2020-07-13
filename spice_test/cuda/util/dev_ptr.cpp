#include <gtest/gtest.h>

#include <spice/cuda/util/dbuffer.h>

#include <spice/cuda/util/dvar.h>
#include <spice/cuda/util/stream.h>


using namespace spice::cuda::util;


void dummy_kernel( cudaStream_t s );


TEST( DevPtr, Ctor )
{
	dbuffer<int> y( 1 );
	ASSERT_EQ( y.size(), 1 );
#pragma warning( push )
#pragma warning( disable : 4389 ) // "signed/unsigned mismatch" size() >= 0
	ASSERT_EQ( y.size_in_bytes(), sizeof( int ) );
#pragma warning( pop )
	ASSERT_NE( y.data(), nullptr );
	ASSERT_EQ( static_cast<nonstd::span<int const>>( y ).size(), 1 );
}

TEST( DevPtr, Resize )
{
	dbuffer<int> x;
	ASSERT_EQ( x.size(), 0 );

	x.resize( 123 );
	ASSERT_EQ( x.size(), 123 );

	x.resize( 1 );
	ASSERT_EQ( x.size(), 1 );
}

TEST( DevPtr, Zero )
{
	dbuffer<float> x( 1 );
	x.zero();
	ASSERT_EQ( x.size(), 1 );

	x.zero_async();
	cudaDeviceSynchronize();
	ASSERT_EQ( x.size(), 1 );
}