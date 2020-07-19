#pragma once

#include <spice/util/assert.h>
#include <spice/util/host_defines.h>

#include <vector>


namespace spice
{
namespace util
{
template <typename Int>
HYBRID Int circidx( Int i, Int size )
{
	spice_assert( size > 0 );
	spice_assert( i >= -size );

	return ( i + size ) % size;
}

template <typename T, typename Container = std::vector<T>>
class circular_buffer
{
public:
	explicit circular_buffer( std::int64_t n = 0 )
	    : _cont( n )
	{
	}

	std::int64_t size() const { return _cont.size(); }

	T & operator[]( std::int64_t i ) { return _cont[circidx( i, size() )]; }
	T const & operator[]( std::int64_t i ) const { return _cont[circidx( i, size() )]; }

	auto begin() { return _cont.begin(); }
	auto begin() const { return _cont.cbegin(); }
	auto end() { return _cont.end(); }
	auto end() const { return _cont.cend(); }

private:
	Container _cont;
};
} // namespace util
} // namespace spice