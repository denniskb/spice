#pragma once

#include <spice/cpu/snn.h>
#include <spice/cuda/util/dbuffer.h>
#include <spice/cuda/util/dvar.h>
#include <spice/cuda/util/stream.h>
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
	     int_ slice_width = -1,
	     int_ n = 1,
	     int_ i = 0 );
	snn( std::vector<int_> const & adj,
	     size_ width,
	     float dt,
	     int_ delay = 1,
	     int_ slice_width = -1,
	     int_ n = 1,
	     int_ i = 0 );
	snn( spice::snn<Model> const & net );

	void step( std::vector<int_> * out_spikes = nullptr ) override;
	void
	step( int_ ** out_dspikes, uint_ ** out_dnum_spikes, std::vector<int_> * out_spikes = nullptr );
	util::stream & sim_stream();

	size_ num_neurons() const override;
	size_ num_synapses() const override;
	std::pair<std::vector<int_>, size_> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	spice::util::soa_t<util::dbuffer, typename Model::neuron> _neurons;
	spice::util::soa_t<util::dbuffer, typename Model::synapse> _synapses;

	struct
	{
		util::dbuffer<int_> edges;
		spice::util::adj_list adj;
		util::dbuffer<uint_> pivots;
		util::dbuffer<int_> ages;
	} _graph;

	struct
	{
		util::dbuffer<int_> ids_data;
		spice::util::span2d<int_> ids;
		util::dbuffer<uint_> counts;

		util::dbuffer<ulong_> history;

		util::dbuffer<int_> updates;
		util::dvar<uint_> num_updates;
	} _spikes;

	int_ const _slice_width;
	int_ const _n;
	int_ const _i;

	util::stream _sim;

	void reserve( size_ num_neurons, size_ max_degree, int_ delay );
};
} // namespace spice::cuda
