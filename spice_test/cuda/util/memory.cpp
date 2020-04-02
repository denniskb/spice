#include <gtest/gtest.h>

#include <spice/cuda/util/memory.h>


using namespace spice::cuda::util;


void write_23( int * p );


TEST( Memory, CudaMalloc )
{
	std::unique_ptr<int, cudaError_t ( * )( void * )> p(
	    static_cast<int *>( cuda_malloc( 4 ) ), cudaFree );

	int x = 4;
	cudaMemcpy( p.get(), &x, 4, cudaMemcpyDefault );

	x = 0;
	cudaMemcpy( &x, p.get(), 4, cudaMemcpyDefault );

	ASSERT_EQ( x, 4 );
}

TEST( Memory, CudaMallocHost )
{
	std::unique_ptr<int, cudaError_t ( * )( void * )> p(
	    static_cast<int *>( cuda_malloc_host( 4 ) ), cudaFree );

	*p = 5;
	ASSERT_EQ( *p, 5 );

	write_23( p.get() );
	EXPECT_EQ( *p, 5 ) << "kernel could've finished";

	cudaDeviceSynchronize();
	ASSERT_EQ( *p, 23 );
}

TEST( Memory, CudaMallocManaged )
{
	std::unique_ptr<int, cudaError_t ( * )( void * )> p(
	    static_cast<int *>( cuda_malloc_managed( 4 ) ), cudaFree );

	*p = 5;
	ASSERT_EQ( *p, 5 );

	write_23( p.get() );
	// EXPECT_EQ(* p, 5) << "kernel could've already finished"; // illegal with managed memory

	cudaDeviceSynchronize();
	ASSERT_EQ( *p, 23 );
}