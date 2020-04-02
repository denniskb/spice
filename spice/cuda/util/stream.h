#pragma once

#include <cuda_runtime.h>


namespace spice::cuda::util
{
class stream
{
public:
	stream() noexcept( false );

	stream( stream const & ) = delete;
	stream & operator=( stream const & ) = delete;

	stream( stream && tmp ) = delete;
	stream & operator=( stream && tmp ) = delete;

	~stream() noexcept;

	static stream & default_stream();

	operator cudaStream_t();

	cudaError_t query();
	void synchronize();
	void wait( cudaEvent_t event );

private:
	cudaStream_t _stream = nullptr;

	explicit stream( cudaStream_t s );
};
} // namespace spice::cuda::util
