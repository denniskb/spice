#include <gtest/gtest.h>

#include <spice/cuda/util/dbuffer.h>
#include <spice/cuda/util/device.h>


using namespace spice::cuda::util;


static auto vec( std::initializer_list<int> l ) { return std::vector<int>( l ); }

TEST( DBuffer, DefaultCtor )
{
	dbuffer<int> x;

	ASSERT_EQ( x.data(), nullptr );
	ASSERT_EQ( x.size(), 0u );
	ASSERT_EQ( x.size_in_bytes(), 0u );
}

TEST( DBuffer, SizeCtor )
{
	dbuffer<int> x( 23 );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x.size(), 23u );
	ASSERT_EQ( x.size_in_bytes(), 23 * sizeof( int ) );

	ASSERT_THROW( dbuffer<int>( 1000'000'000'000 ), std::bad_alloc );
}

TEST( DBuffer, CopyCtor )
{
	{
		dbuffer<int> y( 23 );
		dbuffer<int> x( y );

		ASSERT_NE( x.data(), nullptr );
		ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 23 );
		ASSERT_EQ( x.size(), 23u );
		ASSERT_EQ( x.size_in_bytes(), 23 * sizeof( int ) );
	}

	{
		auto y = vec( { 1, 2, 3, 4, 5 } );
		dbuffer<int> z( y );
		dbuffer<int> x( z );
		y.clear();
		y = x;

		ASSERT_EQ( y, vec( { 1, 2, 3, 4, 5 } ) );
	}
}

TEST( DBuffer, CopyAssign )
{
	{
		dbuffer<int> y( 23 );
		dbuffer<int> x;

		x = y;

		ASSERT_NE( x.data(), nullptr );
		ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 23 );
		ASSERT_EQ( x.size(), 23u );
		ASSERT_EQ( x.size_in_bytes(), 23 * sizeof( int ) );
	}

	{
		auto y = vec( { 1, 2, 3, 4, 5 } );
		dbuffer<int> z( y );
		dbuffer<int> x;
		x = z;
		y.clear();
		y = x;

		ASSERT_EQ( y, vec( { 1, 2, 3, 4, 5 } ) );
	}

	{ // self-assign
		dbuffer<int> x( vec( { 1, 2, 3 } ) );
		auto const p = x.data();
		x = x;

		ASSERT_EQ( x.data(), p );
		ASSERT_EQ( (std::vector<int>)x, vec( { 1, 2, 3 } ) );
	}
}

TEST( DBuffer, ConvCtor )
{
	auto y = vec( { 1, 2, -7 } );
	dbuffer<int> x( y );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 3 );
	ASSERT_EQ( x.size(), 3u );
	ASSERT_EQ( x.size_in_bytes(), 3 * sizeof( int ) );
}

TEST( DBuffer, ConvAssign )
{
	auto y = vec( { 1, 2, -7 } );
	dbuffer<int> x;

	x = y;

	ASSERT_NE( x.data(), nullptr );
	ASSERT_GE( std::abs( std::distance( x.data(), y.data() ) ), 3 );
	ASSERT_EQ( x.size(), 3u );
	ASSERT_EQ( x.size_in_bytes(), 3 * sizeof( int ) );
}

TEST( DBuffer, ConvFunc )
{
	{ // via copy ctor
		auto y = vec( { 1, 2, -7 } );
		dbuffer<int> x( y );

		y.clear();
		y = x;

		ASSERT_EQ( y, vec( { 1, 2, -7 } ) );
	}

	{ // via copy assign
		auto y = vec( { 1, 2, -7 } );
		dbuffer<int> x;

		x = y;
		y.clear();
		y = x;

		ASSERT_EQ( y, vec( { 1, 2, -7 } ) );
	}
}

TEST( DBuffer, Resize )
{
	// resize
	dbuffer<int> x;
	x.resize( 23 );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x.size(), 23u );
	ASSERT_EQ( x.size_in_bytes(), 23 * sizeof( int ) );

	// downsizing maintains contents
	auto const p = x.data();
	auto y = vec( { 1, 2, 3, 4, 5 } );
	x = y;

	ASSERT_EQ( x.data(), p );

	x.resize( 3 );

	ASSERT_EQ( x.data(), p );
	ASSERT_EQ( x.size(), 3u );
	ASSERT_EQ( x.size_in_bytes(), 3 * sizeof( int ) );

	y.clear();
	y = x;

	ASSERT_EQ( y, vec( { 1, 2, 3 } ) );

	// upsizing invalidates contents
	x.resize( 13 );

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x.size(), 13u );
	ASSERT_EQ( x.size_in_bytes(), 13 * sizeof( int ) );

	ASSERT_THROW( x.resize( 1'000'000'000'000 ), std::bad_alloc );
	ASSERT_EQ( x.data(), nullptr );
	ASSERT_EQ( x.size(), 0u );
	ASSERT_EQ( x.size_in_bytes(), 0u );
}

TEST( DBuffer, Zero )
{
	auto y = vec( { 1, 2, 3, 4, 5 } );
	dbuffer<int> x( y );
	x.zero();

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x.size(), 5u );
	ASSERT_EQ( x.size_in_bytes(), 5 * sizeof( int ) );

	y = x;
	ASSERT_EQ( y, vec( { 0, 0, 0, 0, 0 } ) );
}

TEST( DBuffer, ZeroAsync )
{
	auto y = vec( { 1, 2, 3, 4, 5 } );
	dbuffer<int> x( y );
	x.zero_async();
	cudaDeviceSynchronize();

	ASSERT_NE( x.data(), nullptr );
	ASSERT_EQ( x.size(), 5u );
	ASSERT_EQ( x.size_in_bytes(), 5 * sizeof( int ) );

	y = x;
	ASSERT_EQ( y, vec( { 0, 0, 0, 0, 0 } ) );
}

TEST( DBuffer, MultiGPU )
{
	if( device::devices().size() > 1 )
	{
		auto & d = device::active();

		// alloc on gpu0
		cudaSetDevice( 0 );
		dbuffer<int> y( vec( { 1, 2, 3 } ) );

		// gpu1 pushing
		cudaSetDevice( 1 );
		{ // copy ctor
			dbuffer<int> x( y );

			ASSERT_NE( x.data(), y.data() );
			ASSERT_EQ( (std::vector<int>)x, vec( { 1, 2, 3 } ) );
		}
		{ // copy assign
			dbuffer<int> x( 3 );
			x = y;

			ASSERT_NE( x.data(), y.data() );
			ASSERT_EQ( (std::vector<int>)x, vec( { 1, 2, 3 } ) );
		}

		dbuffer<int> x( 3 );
		cudaSetDevice( 0 ); // gpu 0 pulling (only possible with copy assign)
		x = y;

		ASSERT_NE( x.data(), y.data() );
		ASSERT_EQ( (std::vector<int>)x, vec( { 1, 2, 3 } ) );

		d.set();
	}
}