#pragma once

#include <spice/cuda/snn.h>
#include <spice/cuda/util/device.h>
#include <spice/snn.h>
#include <spice/util/span.hpp>

#include <vector>

namespace spice::cuda
{
template <typename Model>
class multi_snn : public ::spice::snn<Model>
{
public:
	multi_snn( spice::util::layout desc, float dt, int_ delay = 1 );
	multi_snn( spice::snn<Model> const & net );

	void step( std::vector<int> * out_spikes = nullptr ) override;
	void sync();

	size_ num_neurons() const override;
	size_ num_synapses() const override;

	std::pair<std::vector<int>, size_> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	std::array<std::optional<cuda::snn<Model>>, util::device::max_devices> _nets;
	struct
	{
		std::unique_ptr<uint_, cudaError_t ( * )( void * )> counts_data;
		nonstd::span<uint_> counts;
	} _spikes;

	multi_snn( float dt, int_ delay );
};
} // namespace spice::cuda