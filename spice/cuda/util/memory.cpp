#include "memory.h"

#include <spice/cuda/util/device.h>
#include <spice/cuda/util/error.h>


namespace spice::cuda::util
{
void * cuda_malloc( size_ n )
{
	void * p = nullptr;
	success_or_throw( cudaMalloc( &p, n ), std::bad_alloc() );

	return p;
}

void * cuda_malloc_host( size_ n )
{
	void * p = nullptr;
	success_or_throw( cudaMallocHost( &p, n ), std::bad_alloc() );

	return p;
}

void * cuda_malloc_managed( size_ n, uint_ flags /* = cudaMemAttachGlobal */ )
{
	void * p = nullptr;
	success_or_throw( cudaMallocManaged( &p, n, flags ), std::bad_alloc() );

	return p;
}
} // namespace spice::cuda::util
