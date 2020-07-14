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
namespace detail
{
inline std::exception err2ex( cudaError_t err )
{
	return std::exception( cudaGetErrorString( err ) );
}

template <typename Ex>
cudaError_t success_or_throw(
    cudaError_t result,
    Ex && exception,
    std::initializer_list<cudaError_t> valid_results = {cudaSuccess} )
{
	if( std::find( valid_results.begin(), valid_results.end(), result ) != valid_results.end() )
		return result;

	cudaGetLastError();
	throw std::forward<Ex>( exception );
}
} // namespace detail

inline cudaError_t success_or_throw( cudaError_t result )
{
	return detail::success_or_throw( result, detail::err2ex( result ), {cudaSuccess} );
}

template <typename Ex>
cudaError_t success_or_throw( cudaError_t result, Ex && exception )
{
	return detail::success_or_throw( result, std::forward<Ex>( exception ), {cudaSuccess} );
}

inline cudaError_t
success_or_throw( cudaError_t result, std::initializer_list<cudaError_t> valid_results )
{
	return detail::success_or_throw( result, detail::err2ex( result ), valid_results );
}
} // namespace util
} // namespace cuda
} // namespace spice
