#pragma once

#include <spice/cuda/snn.h>

#include <vector>

namespace spice::cuda
{
template <typename Model>
class multi_snn
{
public:
	multi_snn( spice::util::layout desc, float dt, int delay = 1 );

	void step( std::vector<int> * out_spikes = nullptr );
	void sync();

	std::size_t num_neurons() const;

private:
	std::vector<snn<Model>> _nets;
	float const _dt;
	int const _delay;
};
} // namespace spice::cuda