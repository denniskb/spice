#include "event.h"

#include <spice/cuda/util/error.h>

#include <utility>

#include <cuda_runtime.h>


namespace spice::cuda::util
{
event::event( uint_ flags ) { success_or_throw( cudaEventCreateWithFlags( &_event, flags ) ); }
event::~event()
{
	if( _event ) // cuda returns invalid_resource_handle on nullptr
	{
		cudaEventSynchronize( _event );
		cudaEventDestroy( _event );
	}
}

event::operator cudaEvent_t() { return _event; }

cudaError_t event::query() const
{
	return success_or_throw( cudaEventQuery( _event ), { cudaSuccess, cudaErrorNotReady } );
}
void event::record( cudaStream_t s ) { success_or_throw( cudaEventRecord( _event, s ) ); }
void event::synchronize() const { success_or_throw( cudaEventSynchronize( _event ) ); }


sync_event::sync_event()
    : event( cudaEventDisableTiming )
{
}


time_event::time_event()
    : event( cudaEventBlockingSync )
{
}
double time_event::elapsed_time( time_event const & since ) const
{
	float ms;
	success_or_throw( cudaEventElapsedTime( &ms, since._event, _event ) );

	return ms * 0.001;
}
} // namespace spice::cuda::util
