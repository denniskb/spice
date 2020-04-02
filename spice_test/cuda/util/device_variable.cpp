#include <gtest/gtest.h>

#include <spice/cuda/util/dev_var.h>

#include <spice/cuda/util/memory.h>


using namespace spice::cuda::util;


TEST( DeviceVariable, Ctor )
{
	// ctor
	dev_var<int> x;
	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x, 0 );

	// ctor value
	dev_var<int> u( 17 );
	ASSERT_EQ( u, 17 );

	// copy from/to host
	double const a = 1.23;
	double b = 0.0;
	dev_var<double> y;
	y = a;
	b = y;
	ASSERT_TRUE( a == b );

	// conversion constructor
	dev_var<int> c( 23 );
	ASSERT_EQ( c, 23 );
}

TEST( DeviceVariable, Zero )
{
	dev_var<int> x;
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

	dev_var<int> y;
	out = -1;
	y.zero_async();
	cudaDeviceSynchronize();
	out = y;
	ASSERT_EQ( out, 0 );
}