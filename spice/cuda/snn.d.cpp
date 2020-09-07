#include "snn.h"

#include <spice/cuda/algorithm.h>
#include <spice/cuda/util/defs.h>
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
int_ snn<Model>::MAX_HISTORY() const
{
	return std::max( this->delay() + 1, 32 );
}

#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced formal parameter 'num_synapses' for certain
                                  // template inst.
template <typename Model>
void snn<Model>::reserve( size_ const num_neurons, size_ const num_synapses, int_ const delay )
{
	spice_assert( num_synapses % num_neurons == 0 );

	// TODO: Hysteresis
	_graph.edges.resize( num_synapses );
	_graph.adj = { num_neurons, num_synapses / num_neurons, _graph.edges.data() };

	_spikes.ids_data.resize( delay * num_neurons );
	_spikes.ids = { _spikes.ids_data.data(), narrow<int>( num_neurons ) };
	_spikes.counts.resize( delay );

	if constexpr( Model::neuron::size > 0 ) _neurons.resize( num_neurons );

	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.resize( num_synapses );

		_spikes.history_data.resize( MAX_HISTORY() * ( ( num_neurons + 31 ) / 32 ) );
		_spikes.history = { _spikes.history_data.data(), narrow<int>( ( num_neurons + 31 ) / 32 ) };
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
    int_ const delay /* = 1 */,
    int_ first /* = 0 */,
    int_ last /* = -1 */ )
    : ::spice::snn<Model>( dt, delay )
    , _first( first )
    , _last( last == -1 ? narrow<int>( desc.size() ) : last )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	reserve( desc.size(), desc.size() * desc.max_degree(), delay );
	generate_rnd_adj_list( desc, _graph.edges.data() );

	upload_meta<Model>( _neurons.data(), _synapses.data() );
	spice::cuda::init<Model>(
	    _first,
	    _last,
	    this->info(),
	    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) } );
	cudaDeviceSynchronize();
}

template <typename Model>
snn<Model>::snn(
    std::vector<int> const & adj,
    size_ width,
    float const dt,
    int_ const delay /* = 1 */,
    int_ first /* = 0 */,
    int_ last /* = -1 */ )
    : ::spice::snn<Model>( dt, delay )
    , _first( first )
    , _last( last == -1 ? narrow<int>( adj.size() / width ) : last )
{
	spice_assert( width > 0 );
	spice_assert( adj.size() % width == 0 );
	spice_assert( width % WARP_SZ == 0 );
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	reserve( adj.size() / width, adj.size(), delay );
	_graph.edges = adj;

	upload_meta<Model>( _neurons.data(), _synapses.data() );
	spice::cuda::init<Model>(
	    _first,
	    _last,
	    this->info(),
	    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) } );
	cudaDeviceSynchronize();
}

template <typename Model>
snn<Model>::snn( spice::snn<Model> const & net )
    : ::spice::snn<Model>( net )
    , _first( 0 )
    , _last( narrow<int>( net.num_neurons() ) )
{
	reserve( net.num_neurons(), net.num_synapses(), net.delay() );

	_graph.edges = net.adj().first;
	_neurons.from_aos( net.neurons() );
	_synapses.from_aos( net.synapses() );

	upload_meta<Model>( _neurons.data(), _synapses.data() );
	cudaDeviceSynchronize();
}


template <typename Model>
void snn<Model>::step( std::vector<int> * out_spikes )
{
	int_ * spikes;
	uint_ * num_spikes;

	step( nullptr, &spikes, &num_spikes, out_spikes );
}

template <typename Model>
void snn<Model>::step(
    cudaEvent_t updt,
    int_ ** out_dspikes,
    uint_ ** out_dnum_spikes,
    std::vector<int> * out_spikes /* = nullptr */ )
{
	this->_step( [&]( int_ const i, float const dt ) {
		// TODO: Improve with co-routines
		auto const _receive = [&]( int iter ) {
			if( iter >= this->delay() )
			{
				receive<Model>(
				    _sim,

				    this->info(),
				    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) },

				    _spikes.ids.row( circidx( iter, this->delay() ) ),
				    _spikes.counts.data() + circidx( iter, this->delay() ),

				    _graph.ages.data(),
				    _spikes.history,
				    MAX_HISTORY(),
				    iter - 1,
				    this->delay(),
				    dt );
			}
		};

		if( this->delay() == 1 ) _receive( i );

		zero_async( _spikes.counts.data() + circidx( i, this->delay() ), _sim );
		if constexpr( Model::synapse::size > 0 ) _spikes.num_updates.zero_async( _sim );

		update<Model>(
		    _sim,
		    updt,

		    _first,
		    _last,
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
		    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) } );

		// delay > 1 => eagerly (and safely) execute next receive()
		if( this->delay() > 1 ) _receive( i + 1 );

		*out_dspikes = _spikes.ids.row( circidx( i, this->delay() ) );
		*out_dnum_spikes = _spikes.counts.data() + circidx( i, this->delay() );

		if( out_spikes )
		{
			uint_ count;
			cudaMemcpy( &count, *out_dnum_spikes, sizeof( uint_ ), cudaMemcpyDefault );
			out_spikes->resize( count );
			cudaMemcpy(
			    out_spikes->data(), *out_dspikes, count * sizeof( int_ ), cudaMemcpyDefault );
		}
	} );
}


template <typename Model>
size_ snn<Model>::num_neurons() const
{
	return _graph.adj.num_nodes();
}
template <typename Model>
size_ snn<Model>::num_synapses() const
{
	return _graph.adj.num_edges();
}
template <typename Model>
std::pair<std::vector<int>, size_> snn<Model>::adj() const
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


template class snn<spice::vogels_abbott>;
template class snn<spice::brunel>;
template class snn<spice::brunel_with_plasticity>;
template class snn<spice::synth>;
} // namespace spice::cuda