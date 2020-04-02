#include "stream.h"

#include <spice/cuda/util/error.h>

#include <utility>


namespace spice::cuda::util
{
stream::stream() { success_or_throw( cudaStreamCreate( &_stream ) ); }
stream::stream( cudaStream_t s )
    : _stream( s )
{
}

stream::~stream()
{
	if( _stream ) // cuda returns invalid_resource_handle on nullptr
	{
		cudaStreamSynchronize( _stream );
		cudaStreamDestroy( _stream );
	}
}


stream & stream::default_stream()
{
	static stream s( nullptr );
	return s;
}


stream::operator cudaStream_t() { return _stream; }


cudaError_t stream::query()
{
	return success_or_throw( cudaStreamQuery( _stream ), {cudaSuccess, cudaErrorNotReady} );
}

void stream::synchronize() { success_or_throw( cudaStreamSynchronize( _stream ) ); }

void stream::wait( cudaEvent_t event )
{
	success_or_throw( cudaStreamWaitEvent( _stream, event, 0 ) );
}
} // namespace spice::cuda::util
