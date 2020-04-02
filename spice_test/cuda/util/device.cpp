#include <gtest/gtest.h>

#include <spice/cuda/util/device.h>


using namespace spice::cuda::util;


TEST( Device, Devices )
{
	int n;
	cudaGetDeviceCount( &n );

	ASSERT_EQ( n, device::devices().size() );

	int d;
	cudaGetDevice( &d );
	ASSERT_TRUE( d == device::active() );

	ASSERT_TRUE( cudaCpuDeviceId == device::cpu );

	if( device::devices().size() == 1 )
		ASSERT_TRUE( device::active() == device::devices( 0 ) );

	ASSERT_FALSE( device::cpu == device::none );
	ASSERT_FALSE( device::active() == device::none );
}

TEST( Device, SetActive )
{
	if( device::devices().size() > 1 )
	{
		for( device & dev : device::devices() )
		{
			dev.set();

			int d;
			cudaGetDevice( &d );
			ASSERT_TRUE( d == dev );
			ASSERT_TRUE( d == device::active() );
			ASSERT_EQ( &dev, &device::active() );
			ASSERT_FALSE( dev == device::none );
		}
	}
}