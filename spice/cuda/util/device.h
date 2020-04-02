#pragma once

#include <spice/util/span.hpp>

#include <cuda_runtime.h>


namespace spice::cuda::util
{
class device
{
public:
	device( device const & ) = delete;
	device & operator=( device const & ) = delete;
	device( device && ) = delete;
	device & operator=( device && ) = delete;

	static nonstd::span<device> devices() noexcept( false );
	static device & devices( std::size_t i );
	static device & active();
	static device const cpu;
	static device const none;

	operator int() const;

	cudaDeviceProp props() const;
	void set();

private:
	int _id;

	constexpr device( int id ) noexcept;
};
} // namespace spice::cuda::util
