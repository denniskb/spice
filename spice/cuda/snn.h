#pragma once

#include <spice/cpu/snn.h>
#include <spice/cuda/util/dbuffer.h>
#include <spice/cuda/util/dvar.h>
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
	snn( spice::util::layout const & desc,
	     float dt,
	     int_ delay = 1,
	     int_ first = 0,
	     int_ last = -1 );
	snn( std::vector<int> const & adj,
	     size_ width,
	     float dt,
	     int_ delay = 1,
	     int_ first = 0,
	     int_ last = -1 );
	snn( spice::snn<Model> const & net );

	void step( std::vector<int> * out_spikes = nullptr ) override;
	void
	step( int_ ** out_dspikes, uint_ ** out_dnum_spikes, std::vector<int> * out_spikes = nullptr );

	size_ num_neurons() const override;
	size_ num_synapses() const override;
	std::pair<std::vector<int>, size_> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	spice::util::soa_t<util::dbuffer, typename Model::neuron> _neurons;
	spice::util::soa_t<util::dbuffer, typename Model::synapse> _synapses;

	struct
	{
		util::dbuffer<int> edges;
		spice::util::adj_list adj;
		util::dbuffer<int> ages;
	} _graph;

	struct
	{
		// TODO: Optimize memory consumption (low-priority)
		util::dbuffer<int> ids_data;
		spice::util::span2d<int> ids;
		util::dbuffer<uint_> counts;

		util::dbuffer<uint_> history_data;
		spice::util::span2d<uint_> history;

		util::dbuffer<int> updates;
		util::dvar<uint_> num_updates;
	} _spikes;

	int_ const _first;
	int_ const _last;

	int_ MAX_HISTORY() const;

	void reserve( size_ num_neurons, size_ max_degree, int_ delay );
};
} // namespace spice::cuda
