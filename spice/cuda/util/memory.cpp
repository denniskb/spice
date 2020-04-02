#include "memory.h"

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/error.h>


namespace spice::cuda::util
{
void * cuda_malloc( std::size_t n )
{
	void * p = nullptr;
	success_or_throw( cudaMalloc( &p, n ), std::bad_alloc() );

	return p;
}

void * cuda_malloc_host( std::size_t n )
{
	void * p = nullptr;
	success_or_throw( cudaMallocHost( &p, n ), std::bad_alloc() );

	return p;
}

void * cuda_malloc_managed( std::size_t n, unsigned flags /* = cudaMemAttachGlobal */ )
{
	void * p = nullptr;
	success_or_throw( cudaMallocManaged( &p, n, flags ), std::bad_alloc() );

	return p;
}
} // namespace spice::cuda::util
