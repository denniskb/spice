#include <gtest/gtest.h>

#include <spice/cuda/util/dvar.h>

#include <spice/cuda/util/memory.h>


using namespace spice::cuda::util;


TEST( DeviceVariable, Ctor )
{
	// ctor
	dvar<int> x;
	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x, 0 );

	// ctor value
	dvar<int> u( 17 );
	ASSERT_EQ( u, 17 );

	// copy from/to host
	double const a = 1.23;
	double b = 0.0;
	dvar<double> y;
	y = a;
	b = y;
	ASSERT_TRUE( a == b );

	// conversion constructor
	dvar<int> c( 23 );
	ASSERT_EQ( c, 23 );
}

TEST( DeviceVariable, Zero )
{
	dvar<int> x;
	int out;

	x.zero();
	out = x;
	ASSERT_EQ( out, 0 );

	out = -1;
	x = 5;
	x.zero();
	out = x;
	ASSERT_EQ( out, 0 );

	x = 7;
	x.zero_async();
	cudaDeviceSynchronize();
	out = x;
	ASSERT_EQ( out, 0 );

	dvar<int> y;
	out = -1;
	y.zero_async();
	cudaDeviceSynchronize();
	out = y;
	ASSERT_EQ( out, 0 );
}