#include <gtest/gtest.h>

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/error.h>


using namespace spice::cuda::util;


// We rely on UVA (letting cuda infer location of memory via ptr address)
// make sure it's supported by all devices.
TEST( Device, UVA )
{
	for( auto & g : device::devices() ) ASSERT_TRUE( g.props().unifiedAddressing );
}

TEST( Device, Devices )
{
	int n;
	cudaGetDeviceCount( &n );

	ASSERT_EQ( n, device::devices().size() );

	int d;
	cudaGetDevice( &d );
	ASSERT_EQ( d, device::active() );

	ASSERT_EQ( device::cpu, cudaCpuDeviceId );

	if( device::devices().size() == 1 ) ASSERT_EQ( device::active(), device::devices( 0 ) );

	ASSERT_NE( device::cpu, device::none );

	for( device & dev : device::devices() )
	{
		ASSERT_NE( dev, device::none );
		ASSERT_NE( dev, device::cpu );
	}
}

TEST( Device, SetActive )
{
	auto & prev = device::active();

	for( device & dev : device::devices() )
	{
		dev.set();

		int d;
		cudaGetDevice( &d );
		ASSERT_EQ( d, dev );
		ASSERT_EQ( d, device::active() );
		ASSERT_EQ( &dev, &device::active() );
	}

	prev.set();
}