#include "event.h"

#include <spice/cuda/util/error.h>

#include <utility>

#include <cuda_runtime.h>


namespace spice::cuda::util
{
event::event() { success_or_throw( cudaEventCreateWithFlags( &_event, cudaEventBlockingSync ) ); }

event::~event()
{
	if( _event ) // cuda returns invalid_resource_handle on nullptr
	{
		cudaEventSynchronize( _event );
		cudaEventDestroy( _event );
	}
}

event::operator cudaEvent_t() { return _event; }

float event::elapsed_time( cudaEvent_t since )
{
	float ms = -1.0f;
	success_or_throw( cudaEventElapsedTime( &ms, since, _event ) );

	return ms;
}

cudaError_t event::query()
{
	return success_or_throw( cudaEventQuery( _event ), { cudaSuccess, cudaErrorNotReady } );
}

void event::record( cudaStream_t s ) { success_or_throw( cudaEventRecord( _event, s ) ); }

void event::synchronize() { success_or_throw( cudaEventSynchronize( _event ) ); }
} // namespace spice::cuda::util
