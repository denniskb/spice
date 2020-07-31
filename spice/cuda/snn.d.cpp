#include "snn.h"

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/device.h>
#include <spice/cuda/util/event.h>
#include <spice/cuda/util/stream.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/assert.h>
#include <spice/util/type_traits.h>

#include <ctime>


using namespace spice::util;
using namespace spice::cuda::util;


namespace spice::cuda
{
template <typename Model>
int snn<Model>::first() const
{
	return narrow_int<int>( _i * num_neurons() / _n );
}

template <typename Model>
int snn<Model>::last() const
{
	return narrow_int<int>( ( _i + 1 ) * num_neurons() / _n );
}

template <typename Model>
int snn<Model>::MAX_HISTORY() const
{
	return std::max( this->delay() + 1, 48 );
}

#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced formal parameter 'num_synapses' for certain
                                  // template inst.
template <typename Model>
void snn<Model>::reserve(
    std::size_t const num_neurons, std::size_t const num_synapses, int const delay )
{
	spice_assert( num_synapses % num_neurons == 0 );

	// TODO: Hysteresis
	_graph.edges.resize( num_synapses );
	_graph.adj = { num_neurons, num_synapses / num_neurons, _graph.edges.data() };

	_spikes.ids_data.resize( delay * num_neurons );
	_spikes.ids = { _spikes.ids_data.data(), narrow_int<int>( num_neurons ) };
	_spikes.counts.resize( delay );

	if constexpr( Model::neuron::size > 0 ) _neurons.resize( num_neurons );

	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.resize( num_synapses );

		_spikes.history_data.resize( MAX_HISTORY() * ( ( num_neurons + 31 ) / 32 ) );
		_spikes.history = {
		    _spikes.history_data.data(), narrow_int<int>( ( num_neurons + 31 ) / 32 ) };
		_spikes.updates.resize( num_neurons );

		_graph.ages.resize( num_neurons );

		_spikes.history_data.zero_async();
		_graph.ages.zero_async();
	}
}
#pragma warning( pop )


template <typename Model>
snn<Model>::snn(
    spice::util::layout const & desc,
    float const dt,
    int const delay /* = 1 */,
    int n /* = 1 */,
    int i /* = 0 */ )
    : ::spice::snn<Model>( dt, delay )
    , _n( n )
    , _i( i )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	reserve( desc.size(), desc.size() * desc.max_degree(), delay );
	generate_rnd_adj_list( desc, _graph.edges.data() );

	upload_meta<Model>( _neurons.data(), _synapses.data() );
	spice::cuda::init<Model>(
	    first(),
	    last(),
	    this->info(),
	    { _graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() ) } );
	cudaDeviceSynchronize();
}

template <typename Model>
snn<Model>::snn( spice::cpu::snn<Model> const & net )
    : ::spice::snn<Model>( net )
{
	reserve( net.num_neurons(), net.num_synapses(), net.delay() );

	_graph.edges = net.graph().first;
	_neurons.from_aos( net.neurons() );
	_synapses.from_aos( net.synapses() );

	upload_meta<Model>( _neurons.data(), _synapses.data() );
	cudaDeviceSynchronize();
}


template <typename Model>
std::size_t snn<Model>::num_neurons() const
{
	return _graph.adj.num_nodes();
}
template <typename Model>
std::size_t snn<Model>::num_synapses() const
{
	return _graph.adj.num_edges();
}
template <typename Model>
std::pair<std::vector<int>, std::size_t> snn<Model>::graph() const
{
	return { _graph.edges, _graph.adj.max_degree() };
}
template <typename Model>
std::vector<typename Model::neuron::tuple_t> snn<Model>::neurons() const
{
	return _neurons.to_aos();
}
template <typename Model>
std::vector<typename Model::synapse::tuple_t> snn<Model>::synapses() const
{
	return _synapses.to_aos();
}


template <typename Model>
void snn<Model>::_step( int const i, float const dt, std::vector<int> * out_spikes )
{
	zero_async( _spikes.counts.data() + circidx( i, this->delay() ) );

	if constexpr( Model::synapse::size > 0 ) _spikes.num_updates.zero_async();

	update<Model>(
	    first(),
	    last(),
	    this->info(),
	    dt,
	    _spikes.ids.row( circidx( i, this->delay() ) ),
	    _spikes.counts.data() + circidx( i, this->delay() ),

	    _spikes.history,
	    _graph.ages.data(),
	    _spikes.updates.data(),
	    _spikes.num_updates.data(),
	    i,
	    this->delay(),
	    MAX_HISTORY(),
	    { _graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() ) } );

	if( i >= this->delay() - 1 )
	{
		receive<Model>(
		    this->info(),
		    { _graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() ) },

		    _spikes.ids.row( circidx( i - this->delay() + 1, this->delay() ) ),
		    _spikes.counts.data() + circidx( i - this->delay() + 1, this->delay() ),

		    _graph.ages.data(),
		    _spikes.history,
		    MAX_HISTORY(),
		    i,
		    this->delay(),
		    dt );
	}

	if( out_spikes )
	{
		unsigned count;
		cudaMemcpy(
		    &count,
		    _spikes.counts.data() + circidx( i, this->delay() ),
		    sizeof( unsigned ),
		    cudaMemcpyDefault );
		out_spikes->resize( count );
		cudaMemcpy(
		    out_spikes->data(),
		    _spikes.ids.row( circidx( i, this->delay() ) ),
		    count * sizeof( unsigned ),
		    cudaMemcpyDefault );
	}
}


template class snn<spice::vogels_abbott>;
template class snn<spice::brunel>;
template class snn<spice::brunel_with_plasticity>;
template class snn<spice::synth>;
} // namespace spice::cuda