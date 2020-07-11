#pragma once

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/error.h>
#include <spice/cuda/util/memory.h>
#include <spice/util/assert.h>
#include <spice/util/span.hpp>
#include <spice/util/type_traits.h>

#include <stdexcept>
#include <type_traits>
#include <vector>


namespace spice
{
namespace cuda
{
namespace util
{
template <typename T>
class dev_ptr
{
public:
	dev_ptr()
	    : _data( nullptr, cudaFree )
	{
	}

	explicit dev_ptr( std::size_t size )
	    : dev_ptr()
	{
		resize( size );
	}

	dev_ptr( dev_ptr && tmp ) = default;
	dev_ptr & operator=( dev_ptr && tmp ) = default;

	operator std::vector<T>() const
	{
		std::vector<T> result( size() );
		cudaMemcpy( result.data(), data(), size_in_bytes(), cudaMemcpyDefault );
		return result;
	}

	// if n > capacity(), deletes the memory
	void resize( std::size_t n )
	{
		if( n > _capacity )
		{
			_data.reset();
			_data.reset( reinterpret_cast<T *>( cuda_malloc( n * sizeof( T ) ) ) );
			_capacity = n;
		}

		_size = n;
	}

	T * data() { return _data.get(); }
	T const * data() const { return _data.get(); }
	std::size_t size() const { return _size; }
	std::size_t size_in_bytes() const { return size() * sizeof( T ); }
	std::size_t capacity() const { return _capacity; }

	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero()
	{
		success_or_throw( cudaMemset( data(), 0, size_in_bytes() ) );
	}
	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero_async( cudaStream_t s = nullptr )
	{
		success_or_throw( cudaMemsetAsync( data(), 0, size_in_bytes(), s ) );
	}

private:
	std::unique_ptr<T, cudaError_t ( * )( void * )> _data;
	std::size_t _size = 0;
	std::size_t _capacity = 0;
};
} // namespace util
} // namespace cuda
} // namespace spice
