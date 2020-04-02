#pragma once

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/dev_ptr.h>


namespace spice::cuda::util
{
template <typename T, typename Parent = dev_ptr<T>>
class dev_var : public Parent
{
public:
	static_assert(
	    std::is_arithmetic_v<T>,
	    "dev_var is only intended for individual primitive variables residing on the device such "
	    "as counters/etc." );

	dev_var() noexcept( false )
	    : Parent( 1 )
	{
		Parent::zero();
	}

	explicit dev_var( T val )
	    : dev_var()
	{
		( *this )[0] = val;
	}

	dev_var & operator=( T val )
	{
		( *this )[0] = val;
		return *this;
	}

	operator T() const { return ( *this )[0]; }


	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero_async( cudaStream_t s = nullptr )
	{
		spice::cuda::zero_async( Parent::data(), s );
		success_or_throw( cudaGetLastError() );
	}
};
} // namespace spice::cuda::util
