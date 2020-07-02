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
	snn( std::size_t num_neurons, float p_connect, float dt, int delay = 1 );
	snn( util::neuron_group const & desc, float dt, int delay = 1 );

	util::adj_list const & graph() const override;
	typename Model::neuron::tuple_t get_neuron( std::size_t i ) const override;
	typename Model::synapse::tuple_t get_synapse( std::size_t i ) const override;

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

	void _step( int i, float dt, std::vector<int> * out_spikes ) override;
};
} // namespace cpu
} // namespace spice
