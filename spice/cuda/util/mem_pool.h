#pragma once

#include <spice/cuda/util/memory.h>

#include <cstddef>
#include <memory>


namespace spice::cuda::util
{
class mem_pool
{
public:
	explicit mem_pool( std::size_t capacity );

	std::size_t capacity() const;
	std::size_t size() const;

	void * alloc( std::size_t size );

private:
	std::unique_ptr<void, cudaError_t ( * )( void * )> _data;
	std::size_t _capacity = 0;
	std::size_t _size = 0;
};
} // namespace spice::cuda::util