#include <gtest/gtest.h>

#include <spice/util/random.h>

#include <random>


using namespace spice::util;

uint_ zerorng() { return 0; }
uint_ maxrng() { return std::numeric_limits<uint_>::max(); }


ulong_ seed()
{
	ulong_ result;
	static std::random_device rd;
	result = rd() | (ulong_)rd() << 32;
	std::cerr << "seed: " << result << std::endl;
	return result;
}

TEST( Random, uniform )
{
	ASSERT_EQ( uniform_left_inc( zerorng ), 0.0f );
	ASSERT_LT( uniform_left_inc( maxrng ), 1.0f );

	ASSERT_GT( uniform_right_inc( zerorng ), 0.0f );
	ASSERT_EQ( uniform_right_inc( maxrng ), 1.0f );

	xoroshiro256ss rng( seed() );

	{
		double m = 0.0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = uniform_left_inc( rng );
			ASSERT_LT( x, 1.0f );
			m += x;
		}
		EXPECT_NEAR( m / 10000, 0.5, 0.01 ) << "Test depends on rng, repeat it.";
	}

	{
		double m = 0.0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = uniform_right_inc( rng );
			ASSERT_GT( x, 0.0f );
			m += x;
		}
		EXPECT_NEAR( m / 10000, 0.5, 0.01 ) << "Test depends on rng, repeat it.";
	}
}

TEST( Random, Exp ) { ASSERT_LT( exprnd( zerorng ), std::numeric_limits<float>::infinity() ); }

TEST( Random, Normal )
{
	xoroshiro128p rng( seed() );

	{
		double m = 0.0;
		for( int_ i = 0; i < 10000; i++ ) m += normrnd( rng );
		EXPECT_NEAR( m / 10000, 0.0, 0.02 ) << "Test depends on rng, repeat it.";
	}

	{
		double m = 0.0;
		for( int_ i = 0; i < 10000; i++ ) m += normrnd( rng, 5 );
		EXPECT_NEAR( m / 10000, 5, 0.02 ) << "Test depends on rng, repeat it.";
	}

	{
		double m = 0.0;
		for( int_ i = 0; i < 10000; i++ ) m += normrnd( rng, -5 );
		EXPECT_NEAR( m / 10000, -5, 0.02 ) << "Test depends on rng, repeat it.";
	}
}

TEST( Random, Binom )
{
	xoroshiro128p rng( seed() );

	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 0 ), 0 );
	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 0.5f ), 0 );
	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 1.0f ), 0 );

	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 1, 0 ), 0 );
	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 1, 1.0f ), 1 );

	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, i, 0 ), 0 );
	for( int_ i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, i, 1.0f ), i );

	{
		uint_ m = 0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = binornd( rng, 1, 0.5f );
			ASSERT_GE( x, 0 );
			ASSERT_LE( x, 1 );
			m += x;
		}
		EXPECT_NEAR( m / 10000.0, 0.5, 0.01 ) << "Test depends on rng, repeat it.";
	}

	{
		uint_ m = 0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = binornd( rng, 100, 0.1f );
			ASSERT_GE( x, 0 );
			ASSERT_LE( x, 100 );
			m += x;
		}
		EXPECT_NEAR( m / 10000.0, 10, 0.1 ) << "Test depends on rng, repeat it.";
	}

	{
		uint_ m = 0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = binornd( rng, 100, 0.5f );
			ASSERT_GE( x, 0 );
			ASSERT_LE( x, 100 );
			m += x;
		}
		EXPECT_NEAR( m / 10000.0, 50, 0.1 ) << "Test depends on rng, repeat it.";
	}

	{
		uint_ m = 0;
		for( int_ i = 0; i < 10000; i++ )
		{
			auto x = binornd( rng, 100, 0.9f );
			ASSERT_GE( x, 0 );
			ASSERT_LE( x, 100 );
			m += x;
		}
		EXPECT_NEAR( m / 10000.0, 90, 0.1 ) << "Test depends on rng, repeat it.";
	}
}