#pragma once

#include <spice/cuda/util/error.h>
#include <spice/cuda/util/memory.h>

#include <type_traits>
#include <vector>


namespace spice
{
namespace cuda
{
namespace util
{
// UNinitialized, managed, type-safe storage on the device.
template <typename T>
class dbuffer
{
public:
	static_assert( std::is_trivially_copyable_v<T>, "copy does not invoke ctors" );

	dbuffer() = default;
	explicit dbuffer( std::size_t size ) { resize( size ); }
	dbuffer( dbuffer const & copy ) { copy_from( copy ); }
	dbuffer( std::vector<T> const & copy ) { copy_from( copy ); }
	dbuffer( dbuffer && tmp ) = default;

	dbuffer & operator=( dbuffer const & rhs )
	{
		copy_from( rhs );
		return *this;
	}
	dbuffer & operator=( std::vector<T> const & rhs )
	{
		copy_from( rhs );
		return *this;
	}
	dbuffer & operator=( dbuffer && tmp ) = default;

	operator std::vector<T>() const
	{
		std::vector<T> result( size() );
		cudaMemcpy( result.data(), data(), size_in_bytes(), cudaMemcpyDefault );
		return result;
	}

	T * data() { return _data.get(); }
	T const * data() const { return _data.get(); }
	std::size_t size() const { return _size; }
	std::size_t size_in_bytes() const { return size() * sizeof( T ); }

	// if n > size(), deletes the memory! (basic exception guarantee)
	void resize( std::size_t n )
	{
		if( n > size() )
		{
			_data.reset();
			_size = 0;
			_data.reset( reinterpret_cast<T *>( cuda_malloc( n * sizeof( T ) ) ) );
		}

		_size = n;
	}

	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero()
	{
		zero_async();
		cudaDeviceSynchronize();
	}
	template <typename U = T>
	std::enable_if_t<std::is_arithmetic_v<U>> zero_async( cudaStream_t s = nullptr )
	{
		success_or_throw( cudaMemsetAsync( data(), 0, size_in_bytes(), s ) );
	}

private:
	std::unique_ptr<T, cudaError_t ( * )( void * )> _data{ nullptr, cudaFree };
	std::size_t _size = 0;

	template <typename Cont>
	void copy_from( Cont const & cont )
	{
		if( data() == cont.data() && size() == cont.size() ) return;

		resize( cont.size() );
		cudaMemcpy( data(), cont.data(), size_in_bytes(), cudaMemcpyDefault );
	}
};
} // namespace util
} // namespace cuda
} // namespace spice
