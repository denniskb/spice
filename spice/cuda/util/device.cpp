#include "device.h"

#include <spice/cuda/util/error.h>
#include <spice/util/assert.h>

#include <iostream>
#include <vector>


using namespace spice::util;


namespace spice::cuda::util
{
nonstd::span<device> device::devices()
{
	static std::array<device, 8> _devices{ 0, 1, 2, 3, 4, 5, 6, 7 };
	static int const n = [] {
		int i = 0;
		success_or_throw( cudaGetDeviceCount( &i ) );
		spice_assert( i <= _devices.size(), "spice does not support more than 8 gpus per node." );
		return i;
	}();

	return nonstd::span( _devices.data(), n );
}

device & device::devices( std::size_t i ) { return devices()[i]; }

device & device::active()
{
	int d;
	success_or_throw( cudaGetDevice( &d ) );

	return devices( d );
}

device const device::cpu = device( cudaCpuDeviceId );
device const device::none = device( -2 );


device::operator int() const { return _id; }


cudaDeviceProp device::props() const
{
	cudaDeviceProp props{};
	success_or_throw( cudaGetDeviceProperties( &props, _id ) );

	return props;
}

void device::set() { success_or_throw( cudaSetDevice( _id ) ); }


constexpr device::device( int id ) noexcept
    : _id( id )
{
}
} // namespace spice::cuda::util
