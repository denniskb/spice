#include <gtest/gtest.h>

#include <spice/util/numeric.h>


using namespace spice::util;


TEST( KahanSum, Ctor )
{
	kahan_sum<float> x;
	ASSERT_EQ( (float)x, 0.0f );
}

TEST( KahanSum, Add )
{
	int const ITER = 1000;
	float const DELTA = 0.001f;

	kahan_sum<float> ksum;
	float deltas = 0;
	for( int i = 0; i < ITER; i++ )
	{
		ASSERT_FLOAT_EQ( (float)ksum, i * DELTA );

		float delta = ksum.add( DELTA );
		deltas += delta;

		ASSERT_NEAR( delta, DELTA, DELTA / 100 );
	}

	// would fail with 'float ksum':
	ASSERT_FLOAT_EQ( (float)ksum, 1 );
	ASSERT_FLOAT_EQ( deltas, 1 );
}