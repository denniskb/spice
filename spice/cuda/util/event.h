#pragma once

#include <spice/util/stdint.h>

#include <cuda_runtime.h>


namespace spice::cuda::util
{
class event
{
public:
	event( event const & ) = delete;
	event & operator=( event const & ) = delete;
	event( event && tmp ) = delete;
	event & operator=( event && tmp ) = delete;

	virtual ~event() noexcept;

	operator cudaEvent_t();

	cudaError_t query() const;
	void record( cudaStream_t s = nullptr );
	void synchronize() const;

protected:
	cudaEvent_t _event = nullptr;
	explicit event( uint_ flags ) noexcept( false );
};

class sync_event : public event
{
public:
	sync_event();
};

class time_event : public event
{
public:
	time_event();

	// in seconds
	double elapsed_time( time_event const & since ) const;
};
} // namespace spice::cuda::util
