#pragma once

#include <spice/cpu/backend.h>
#include <spice/snn.h>
#include <spice/util/adj_list.h>
#include <spice/util/meta.h>
#include <spice/util/span.hpp>

#include <optional>
#include <vector>


namespace spice
{
namespace cpu
{
template <typename Model>
class snn : public ::spice::snn<Model>
{
public:
	snn( util::layout const & desc, float dt, int delay = 1 );

	void step( std::vector<int> * out_spikes = nullptr ) override;

	std::size_t num_neurons() const override;
	std::size_t num_synapses() const override;
	// (edges, width)
	std::pair<std::vector<int>, std::size_t> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	std::optional<std::vector<typename Model::neuron::tuple_t>> _neurons;
	std::optional<std::vector<typename Model::synapse::tuple_t>> _synapses;
	struct
	{
		std::vector<int> edges;
		util::adj_list adj;
	} _graph;

	struct
	{
		std::vector<int> ids;
		std::vector<std::size_t> counts;
		std::optional<std::vector<std::vector<bool>>> flags;
	} _spikes;

	backend _backend;
};
} // namespace cpu
} // namespace spice
