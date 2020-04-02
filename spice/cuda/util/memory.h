#pragma once

#include <cstddef>

#include <cuda_runtime.h>


namespace spice::cuda::util
{
void * cuda_malloc( std::size_t n );
void * cuda_malloc_host( std::size_t n );
void * cuda_malloc_managed( std::size_t n, unsigned flags = cudaMemAttachGlobal );
} // namespace spice::cuda::util
