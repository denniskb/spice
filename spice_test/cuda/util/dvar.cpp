#include <gtest/gtest.h>

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/dvar.h>


using namespace spice::cuda::util;


TEST( DVar, Ctor )
{
	dvar<int> x;

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x, 0 );
}

TEST( DVar, ConvCtor )
{
	dvar<int> x( 5 );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x, 5 );
}

TEST( DVar, CopyCtor )
{
	dvar<int> y( 5 );
	dvar<int> x( y );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 1 );
	ASSERT_EQ( x, 5 );
}

TEST( DVar, CopyAssign )
{
	dvar<int> y( 5 );
	dvar<int> x;
	x = y;

	ASSERT_NE( x.data(), nullptr );
	ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 1 );
	ASSERT_EQ( x, 5 );
}

TEST( DVar, Assign )
{
	dvar<int> x;
	x = 5;

	ASSERT_EQ( x, 5 );
}

TEST( DVar, MultiGPU )
{
	if( device::devices().size() > 1 )
	{
		auto & d = device::active();

		cudaSetDevice( 0 );
		dvar<int> y( 5 );

		cudaSetDevice( 1 );
		{
			dvar<int> x( y );
			ASSERT_EQ( x, 5 );
		}
		{
			dvar<int> x;
			x = y;
			ASSERT_EQ( x, 5 );
		}

		dvar<int> x;
		cudaSetDevice( 0 );
		x = y;

		ASSERT_EQ( x, 5 );

		d.set();
	}
}