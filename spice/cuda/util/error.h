#pragma once

#include <algorithm>
#include <exception>
#include <initializer_list>

#include <cuda_runtime.h>


namespace spice
{
namespace cuda
{
namespace util
{
template <typename Ex>
cudaError_t success_or_throw(
    cudaError_t result,
    Ex exception,
    std::initializer_list<cudaError_t> valid_results = {static_cast<cudaError_t>( 0 )} )
{
	if( std::find( valid_results.begin(), valid_results.end(), result ) != valid_results.end() )
		return result;

	throw exception;
}

template <typename cudaError_t>
inline cudaError_t success_or_throw(
    cudaError_t result,
    std::initializer_list<cudaError_t> valid_results = {static_cast<cudaError_t>( 0 )} )
{
	if( std::find( valid_results.begin(), valid_results.end(), result ) != valid_results.end() )
		return result;

	throw std::exception( cudaGetErrorString( result ) );
}
} // namespace util
} // namespace cuda
} // namespace spice
