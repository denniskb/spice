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
void * cuda_malloc( size_ n );
void * cuda_malloc_host( size_ n );
void * cuda_malloc_managed( size_ n, uint_ flags = cudaMemAttachGlobal );
} // namespace util
} // namespace cuda
} // namespace spice
