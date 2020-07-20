#include <gtest/gtest.h>

#include "model.h"

#include <spice/cpu/snn.h>


using namespace spice;


std::size_t const N = 1000;
float const P = 0.1f;
float const DT = 0.0001f;
int const DELAY = 15;


TEST_ALL_MODELS( SNN );

TYPED_TEST( SNN, Ctor )
{
	{
		cpu::snn<TypeParam> x( { N, P }, DT );

		ASSERT_EQ( x.num_neurons(), N );
		ASSERT_EQ( x.dt(), DT );
		ASSERT_EQ( x.delay(), 1 );
	}

	{
		cpu::snn<TypeParam> x( { N, P }, DT, DELAY );

		ASSERT_EQ( x.num_neurons(), N );
		ASSERT_EQ( x.dt(), DT );
		ASSERT_EQ( x.delay(), DELAY );
	}
}