#include <gtest/gtest.h>

#include <spice/util/random.h>

#include <random>


using namespace spice::util;

unsigned zerorng() { return 0; }
unsigned maxrng() { return UINT_MAX; }


unsigned long long seed()
{
	unsigned long long result;
	std::random_device rd;
	result = rd() | (unsigned long long)rd() << 32;
	std::cerr << "seed: " << result << std::endl;
	return result;
}

TEST( Random, uniform )
{
	ASSERT_EQ( uniform_inc( zerorng ), 0.0f );
	ASSERT_EQ( uniform_inc( maxrng ), 1.0f );

	ASSERT_GT( uniform_ex( zerorng ), 0.0f );
	ASSERT_LT( uniform_ex( maxrng ), 1.0f );

	ASSERT_EQ( uniform_left_inc( zerorng ), 0.0f );
	ASSERT_LT( uniform_left_inc( maxrng ), 1.0f );

	ASSERT_GT( uniform_right_inc( zerorng ), 0.0f );
	ASSERT_EQ( uniform_right_inc( maxrng ), 1.0f );
}

TEST( Random, Exp ) { ASSERT_LT( exprnd( zerorng ), std::numeric_limits<float>::infinity() ); }

TEST( Random, Normal )
{
	xoroshiro128p rng( seed() );

	{
		double m = 0.0;
		for( int i = 0; i < 10000; i++ ) m += normrnd( rng );
		EXPECT_NEAR( m / 10000, 0.0, 0.02 );
	}

	{
		double m = 0.0;
		for( int i = 0; i < 10000; i++ ) m += normrnd( rng, 5 );
		EXPECT_NEAR( m / 10000, 5, 0.02 );
	}

	{
		double m = 0.0;
		for( int i = 0; i < 10000; i++ ) m += normrnd( rng, -5 );
		EXPECT_NEAR( m / 10000, -5, 0.02 );
	}
}

TEST( Random, Binom )
{
	xoroshiro128p rng( seed() );

	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 0 ), 0 );
	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 0.5f ), 0 );
	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 0, 1.0f ), 0 );

	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 1, 0 ), 0 );
	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, 1, 1.0f ), 1 );

	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, i, 0 ), 0 );
	for( int i = 0; i < 100; i++ ) ASSERT_EQ( binornd( rng, i, 1.0f ), i );

	{
		unsigned m = 0;
		for( int i = 0; i < 10000; i++ ) m += binornd( rng, 1, 0.5f );
		EXPECT_NEAR( m / 10000.0, 0.5, 0.01 );
	}

	{
		unsigned m = 0;
		for( int i = 0; i < 10000; i++ ) m += binornd( rng, 100, 0.1f );
		EXPECT_NEAR( m / 10000.0, 10, 0.1 );
	}

	{
		unsigned m = 0;
		for( int i = 0; i < 10000; i++ ) m += binornd( rng, 100, 0.5f );
		EXPECT_NEAR( m / 10000.0, 50, 0.1 );
	}

	{
		unsigned m = 0;
		for( int i = 0; i < 10000; i++ ) m += binornd( rng, 100, 0.9f );
		EXPECT_NEAR( m / 10000.0, 90, 0.1 );
	}
}