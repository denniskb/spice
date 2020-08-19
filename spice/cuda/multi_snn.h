#pragma once

#include <spice/cuda/snn.h>
#include <spice/snn.h>

#include <vector>

namespace spice::cuda
{
template <typename Model>
class multi_snn : public ::spice::snn<Model>
{
public:
	multi_snn( spice::util::layout desc, float dt, int delay = 1 );
	// cpu::snn->cuda::snn converting ctor

	void step( std::vector<int> * out_spikes = nullptr ) override;
	void sync();

	std::size_t num_neurons() const override;
	std::size_t num_synapses() const override;

	std::pair<std::vector<int>, std::size_t> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	std::vector<cuda::snn<Model>> _nets;
};
} // namespace spice::cuda