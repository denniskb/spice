#pragma once

#include <cuda_runtime.h>


namespace spice::cuda::util
{
class event
{
public:
	event() noexcept( false );

	event( event const & ) = delete;
	event & operator=( event const & ) = delete;

	event( event && tmp ) = delete;
	event & operator=( event && tmp ) = delete;

	~event() noexcept;

	operator cudaEvent_t();

	float elapsed_time( cudaEvent_t since );
	cudaError_t query();
	void record( cudaStream_t s = nullptr );
	void synchronize();

private:
	cudaEvent_t _event = nullptr;
};
} // namespace spice::cuda::util
