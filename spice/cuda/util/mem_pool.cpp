#include "mem_pool.h"

#include <spice/util/assert.h>


namespace spice::cuda::util
{
mem_pool::mem_pool( std::size_t capacity )
    : _data( cuda_malloc_managed( capacity ), cudaFree )
    , _capacity( capacity )
{
}

std::size_t mem_pool::capacity() const { return _capacity; }
std::size_t mem_pool::size() const { return _size; }

void * mem_pool::alloc( std::size_t n )
{
	spice_assert( size() + n <= capacity(), "out of memory" );

	auto result = reinterpret_cast<char *>( _data.get() ) + size();
	_size += n;

	return result;
}
} // namespace spice::cuda::util