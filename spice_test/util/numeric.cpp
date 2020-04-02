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
	for( int i = 0; i < ITER; i++ )
	{
		ASSERT_FLOAT_EQ( (float)ksum, i * DELTA );
		ASSERT_NEAR( ksum.add( DELTA ), DELTA, DELTA / 100 );
	}

	// would fail with 'float ksum':
	ASSERT_FLOAT_EQ( (float)ksum, 1 );
}