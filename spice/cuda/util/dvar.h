#pragma once

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/dbuffer.h>


namespace spice::cuda::util
{
template <typename T>
class dvar
{
public:
	static_assert(
	    std::is_arithmetic_v<T>,
	    "dvar is only intended for individual primitive variables residing on the device" );

	dvar() { _data.zero(); }
	dvar( T x )
	    : dvar()
	{
		copy_from( x );
	}

	dvar & operator=( T x )
	{
		copy_from( x );
		return *this;
	}

	operator T() const
	{
		T x;
		cudaMemcpy( &x, data(), sizeof( T ), cudaMemcpyDefault );
		return x;
	}

	T * data() { return _data.data(); }
	T const * data() const { return _data.data(); }

	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero()
	{
		zero_async();
		cudaDeviceSynchronize();
	}

	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero_async( cudaStream_t s = nullptr )
	{
		// TODO: Experiment with memcpy instead of kernel call (overlap)
		spice::cuda::zero_async( _data.data(), s );
		success_or_throw( cudaGetLastError() );
	}

private:
	dbuffer<T> _data{ 1 };
	void copy_from( T x ) { cudaMemcpy( _data.data(), &x, sizeof( T ), cudaMemcpyDefault ); }
};
} // namespace spice::cuda::util