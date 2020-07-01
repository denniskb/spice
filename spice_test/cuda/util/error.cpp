#include <gtest/gtest.h>

#include <spice/cuda/util/error.h>


using namespace spice::cuda::util;


TEST( Error, ThrowOnError )
{
	ASSERT_THROW( success_or_throw( cudaSuccess, {} ), std::exception );

	ASSERT_NO_THROW( success_or_throw( cudaSuccess ) );
	ASSERT_THROW(
	    success_or_throw( cudaSuccess, {cudaErrorNoDevice, cudaErrorAddressOfConstant} ),
	    std::exception );

	for( std::size_t i = 0; i < 100; i++ )
	{
		auto err = static_cast<cudaError>( i );

		ASSERT_NO_THROW( success_or_throw( err, {err} ) );

		if( err != cudaSuccess )
		{
			ASSERT_THROW( success_or_throw( err ), std::exception );
			ASSERT_THROW(
			    success_or_throw( err, std::invalid_argument( "" ) ), std::invalid_argument );
		}
	}

	ASSERT_NO_THROW( success_or_throw( cudaSuccess, {cudaSuccess, cudaErrorNoDevice} ) );
	ASSERT_NO_THROW( success_or_throw( cudaSuccess, {cudaErrorNoDevice, cudaSuccess} ) );
}