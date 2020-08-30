#include <gtest/gtest.h>

#include <cuda_runtime.h>


int main( int argc, char ** argv )
{
	::testing::InitGoogleTest( &argc, argv );
	// cudaSetDevice( 1 );
	return RUN_ALL_TESTS();
}
