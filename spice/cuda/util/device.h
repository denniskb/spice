#pragma once

#include <spice/util/span.hpp>
#include <spice/util/stdint.h>

#include <cuda_runtime.h>


namespace spice
{
namespace cuda
{
namespace util
{
class device
{
public:
	device( device const & ) = delete;
	device & operator=( device const & ) = delete;
	device( device && ) = delete;
	device & operator=( device && ) = delete;

	static constexpr size_ max_devices = 8;
	static nonstd::span<device> devices() noexcept( false );
	static device & devices( size_ i );
	static device & active();
	static device const cpu;
	static device const none;

	operator int_() const;

	cudaDeviceProp props() const;
	void set();
	void synchronize();

private:
	int_ _id;

	device( int_ id ) noexcept;
};
} // namespace util
} // namespace cuda
} // namespace spice
