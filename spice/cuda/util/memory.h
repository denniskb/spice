#pragma once

#include <cstddef>
#include <spice/util/stdint.h>

#include <cuda_runtime.h>


namespace spice
{
namespace cuda
{
namespace util
{
void * cuda_malloc( std::size_t n );
void * cuda_malloc_host( std::size_t n );
void * cuda_malloc_managed( std::size_t n, uint_ flags = cudaMemAttachGlobal );
} // namespace util
} // namespace cuda
} // namespace spice
