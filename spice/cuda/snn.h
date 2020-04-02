#pragma once

#include <spice/cpu/snn.h>
#include <spice/cuda/util/dev_ptr.h>
#include <spice/cuda/util/dev_var.h>
#include <spice/snn.h>
#include <spice/util/circular_buffer.h>
#include <spice/util/meta.h>
#include <spice/util/span.hpp>
#include <spice/util/span2d.h>


namespace spice::cuda
{
template <typename Model>
class snn : public ::spice::snn<Model>
{
public:
	snn() = default;
	snn( std::size_t num_neurons, float p_connect, float dt, int delay = 1 );
	snn( spice::util::neuron_group const & desc, float dt, int delay = 1 );
	snn( spice::cpu::snn<Model> const & net );

	void init( std::size_t num_neurons, float p_connect, float dt, int delay = 1 );
	void init( spice::util::neuron_group const & desc, float dt, int delay = 1 );

	/**
	 * Sets this network as active. Only one network can be active at any given time. Calling step()
	 * is only legal on the active network, and results in undefined behavior otherwise. By default,
	 * the most recently instantiated network is set as active.
	 */
	void set();

	spice::util::adj_list const & graph() const override;
	typename Model::neuron::tuple_t get_neuron( std::size_t i ) const override;
	typename Model::synapse::tuple_t get_synapse( std::size_t i ) const override;

private:
	spice::util::soa_t<util::dev_ptr, typename Model::neuron> _neurons;
	spice::util::soa_t<util::dev_ptr, typename Model::synapse> _synapses;

	struct
	{
		util::dev_ptr<int> edges;
		spice::util::adj_list adj;
		util::dev_ptr<int> ages;
	} _graph;

	struct
	{
		// TODO: Optimize memory consumption (low-priority)
		util::dev_ptr<int> ids_data;
		spice::util::span2d<int> ids;
		util::dev_ptr<unsigned> counts;

		util::dev_ptr<unsigned> history_data;
		spice::util::span2d<unsigned> history;

		util::dev_ptr<int> updates;
		util::dev_var<unsigned> num_updates;
	} _spikes;

	int MAX_HISTORY() const;

	void reserve( std::size_t num_neurons, std::size_t max_degree, int delay );

	void _step( int i, float dt, std::vector<int> * out_spikes ) override;
};
} // namespace spice::cuda
