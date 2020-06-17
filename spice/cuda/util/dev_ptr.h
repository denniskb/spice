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
	using value_type = T;
	using reference = T &;
	using const_reference = T const &;
	using iterator = T *;
	using const_iterator = T const *;
	using difference_type = std::size_t;
	using size_type = std::size_t;

	dev_ptr()
	    : _data( nullptr, cudaFree )
	{
	}

	explicit dev_ptr( std::size_t size )
	    : dev_ptr()
	{
		resize( size );
	}
	explicit dev_ptr( std::vector<T> const & copy )
	    : dev_ptr()
	{
		*this = copy;
	}
	// TODO: Test
	dev_ptr( T const * src, std::size_t size )
	    : dev_ptr( size )
	{
		success_or_throw( cudaMemcpy( data(), src, size * sizeof( T ), cudaMemcpyDefault ) );
	}

	dev_ptr( dev_ptr const & copy )
	    : dev_ptr( copy.data(), copy.size() )
	{
	}
	dev_ptr & operator=( dev_ptr const & rhs )
	{
		( *this ) = dev_ptr( rhs );
		return *this;
	}
	dev_ptr & operator=( std::vector<T> const & rhs )
	{
		resize( rhs.size() );
		success_or_throw(
		    cudaMemcpy( data(), rhs.data(), rhs.size() * sizeof( T ), cudaMemcpyHostToDevice ) );

		return *this;
	}

	dev_ptr( dev_ptr && tmp ) = default;
	dev_ptr & operator=( dev_ptr && tmp ) = default;

	// if n > capacity(), deletes the memory
	void resize( std::size_t n )
	{
		if( n > _capacity )
		{
			_data.reset();
			_data.reset( reinterpret_cast<T *>( cuda_malloc_managed( n * sizeof( T ) ) ) );
			_capacity = n;
		}

		_size = n;
	}

	T * data() { return _data.get(); }
	T const * data() const { return _data.get(); }
	std::size_t size() const { return _size; }
	std::size_t size_in_bytes() const { return size() * sizeof( T ); }
	std::size_t capacity() const { return _capacity; }

	operator nonstd::span<T const>() const { return nonstd::span<T const>( data(), size() ); }

	T & operator[]( std::size_t i )
	{
		spice_assert( i >= 0 && i < size(), "index ouf of bounds" );
		return data()[i];
	}
	T const & operator[]( std::size_t i ) const
	{
		spice_assert( i >= 0 && i < size(), "index out of bounds" );
		return data()[i];
	}

	T * begin() { return data(); }
	T const * begin() const { return data(); }
	T const * cbegin() const { return data(); }
	T * end() { return data() + size(); }
	T const * end() const { return data() + size(); }
	T const * cend() const { return data() + size(); }

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

	// clang-format off
	[[deprecated]] dev_ptr & accessed_by( device const & dev, bool flag = true )
	{
		if( dev != device::none )
		{
			if( flag )
				success_or_throw(
				    cudaMemAdvise( data(), size_in_bytes(), cudaMemAdviseSetAccessedBy, dev ) );
			else
				success_or_throw(
				    cudaMemAdvise( data(), size_in_bytes(), cudaMemAdviseUnsetAccessedBy, dev ) );
		}
		else
		{
			success_or_throw( cudaMemAdvise(
			    data(), size_in_bytes(), cudaMemAdviseUnsetAccessedBy, device::cpu ) );

			for( auto & d : device::devices() )
				success_or_throw(
				    cudaMemAdvise( data(), size_in_bytes(), cudaMemAdviseUnsetAccessedBy, d ) );
		}

		return *this;
	}
	dev_ptr & attach( cudaStream_t s, unsigned flags = cudaMemAttachSingle )
	{
		// TODO: Does this expect size() or size_in_bytes()?
		success_or_throw( cudaStreamAttachMemAsync<T>( s, data(), size_in_bytes(), flags ) );
		return *this;
	}
	[[deprecated]] dev_ptr & location( device const & dev )
	{
		if( dev != device::none )
			success_or_throw(
			    cudaMemAdvise( data(), size_in_bytes(), cudaMemAdviseSetPreferredLocation, dev ) );
		else
			success_or_throw( cudaMemAdvise(
			    data(), size_in_bytes(), cudaMemAdviseUnsetPreferredLocation, dev ) );

		return *this;
	}
	[[deprecated]] dev_ptr & prefetch( device const & dev, cudaStream_t s = nullptr )
	{
		success_or_throw( cudaMemPrefetchAsync( data(), size_in_bytes(), dev, s ) );
		return *this;
	}
	dev_ptr & read_mostly( bool flag = true )
	{
		if( flag )
			success_or_throw( cudaMemAdvise(
			    data(), size_in_bytes(), cudaMemAdviseSetReadMostly, device::none ) );
		else
			success_or_throw( cudaMemAdvise(
			    data(), size_in_bytes(), cudaMemAdviseUnsetReadMostly, device::none ) );

		return *this;
	}
	// clang-format on

private:
	std::unique_ptr<T, cudaError_t ( * )( void * )> _data;
	std::size_t _size = 0;
	std::size_t _capacity = 0;
};

template <typename T>
bool operator==( dev_ptr<T> const & lhs, dev_ptr<T> const & rhs )
{
	return lhs.data() == rhs.data() ||
	       lhs.size() == rhs.size() && std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}

template <typename T>
bool operator==( dev_ptr<T> const & lhs, std::vector<T> const & rhs )
{
	return spice::util::narrow_cast<std::size_t>( lhs.size() ) == rhs.size() &&
	       std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}
template <typename T>
bool operator==( std::vector<T> const & lhs, dev_ptr<T> const & rhs )
{
	return rhs == lhs;
}

template <typename T>
bool operator!=( dev_ptr<T> const & lhs, dev_ptr<T> const & rhs )
{
	return lhs.size() != rhs.size() || !std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}

template <typename T>
bool operator!=( dev_ptr<T> const & lhs, std::vector<T> const & rhs )
{
	return spice::util::narrow_cast<std::size_t>( lhs.size() ) != rhs.size() ||
	       !std::equal( lhs.begin(), lhs.end(), rhs.begin() );
}
template <typename T>
bool operator!=( std::vector<T> const & lhs, dev_ptr<T> const & rhs )
{
	return rhs != lhs;
}
} // namespace util
} // namespace cuda
} // namespace spice
