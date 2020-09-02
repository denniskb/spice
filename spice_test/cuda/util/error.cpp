#include <gtest/gtest.h>

#include <spice/cuda/util/error.h>
#include <spice/util/stdint.h>


using namespace spice::cuda::util;


TEST( Error, ThrowOnError )
{
	ASSERT_THROW( success_or_throw( cudaSuccess, {} ), std::exception );

	ASSERT_NO_THROW( success_or_throw( cudaSuccess ) );
	ASSERT_THROW(
	    success_or_throw( cudaSuccess, { cudaErrorNoDevice, cudaErrorAddressOfConstant } ),
	    std::exception );

	for( size_ i = 0; i < 100; i++ )
	{
		auto err = static_cast<cudaError>( i );

		ASSERT_NO_THROW( success_or_throw( err, { err } ) );

		if( err != cudaSuccess )
		{
			ASSERT_THROW( success_or_throw( err ), std::exception );
			ASSERT_THROW(
			    success_or_throw( err, std::invalid_argument( "" ) ), std::invalid_argument );
		}
	}

	ASSERT_NO_THROW( success_or_throw( cudaSuccess, { cudaSuccess, cudaErrorNoDevice } ) );
	ASSERT_NO_THROW( success_or_throw( cudaSuccess, { cudaErrorNoDevice, cudaSuccess } ) );
}

TEST( Error, ClearAfterThrow )
{
	ASSERT_EQ( cudaGetLastError(), cudaSuccess );
	int_ * p;
	ASSERT_THROW( success_or_throw( cudaMalloc( &p, 1'000'000'000'000 ) ), std::exception );
	ASSERT_EQ( cudaGetLastError(), cudaSuccess );
}