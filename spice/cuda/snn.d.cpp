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
int snn<Model>::MAX_HISTORY() const
{
	return std::max( this->delay() + 1, 48 );
}

template <typename Model>
void snn<Model>::reserve(
    std::size_t const num_neurons, std::size_t const max_degree, int const delay )
{
	auto num_synapses = num_neurons * max_degree;
	if( num_synapses > _graph.edges.capacity() )
	{
		// Hysteresis (TODO: Make generic (compute avg. connect. inside neuron group))
		double const p = std::is_same_v<Model, vogels_abbott> ? 0.02 : 0.05;
		num_synapses = narrow_cast<std::size_t>(
		    ( std::sqrt( 15.0 / p / num_neurons ) + 1.0 ) * num_synapses );
	}

	_spikes.ids_data.resize( delay * num_neurons );
	_spikes.ids = {_spikes.ids_data.data(), narrow_int<int>( num_neurons )};
	_spikes.counts.resize( delay );

	_graph.edges.resize( num_synapses );
	_graph.adj = {num_neurons, max_degree, _graph.edges.data()};

	if constexpr( Model::neuron::size > 0 )
		_neurons.resize( num_neurons );

	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.resize( num_synapses );

		_spikes.history_data.resize( MAX_HISTORY() * ( ( num_neurons + 31 ) / 32 ) );
		_spikes.history = {_spikes.history_data.data(),
		                   narrow_int<int>( ( num_neurons + 31 ) / 32 )};
		_spikes.updates.resize( num_neurons );

		_graph.ages.resize( num_neurons );

		_spikes.history_data.zero_async();
		_graph.ages.zero_async();
	}
}

template <typename Model>
snn<Model>::snn(
    std::size_t const num_neurons,
    float const p_connect,
    float const dt,
    int const delay /* = 1 */ )
    : snn( spice::util::neuron_group( num_neurons, p_connect ), dt, delay )
{
	spice_assert( num_neurons >= 0 && num_neurons < std::numeric_limits<int>::max() );
	spice_assert( p_connect >= 0.0f && p_connect <= 1.0f );
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );
}

template <typename Model>
snn<Model>::snn( spice::util::neuron_group const & desc, float const dt, int const delay /* = 1 */ )
    : ::spice::snn<Model>( dt, delay )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	init( desc, dt, delay );
	cudaDeviceSynchronize();
}

template <typename Model>
snn<Model>::snn( spice::cpu::snn<Model> const & net )
    : ::spice::snn<Model>( net )
{
	reserve( net.num_neurons(), net.graph().max_degree(), net.delay() );

	cudaMemcpy(
	    _graph.edges.data(), net.graph().edges(), net.num_synapses() * 4, cudaMemcpyHostToDevice );
	_graph.edges.read_mostly();

	if constexpr( Model::neuron::size > 0 )
		for( std::size_t i = 0; i < net.num_neurons(); i++ )
			map( _neurons, net.get_neuron( i ), [i]( auto & dptr, auto elem ) { dptr[i] = elem; } );

	if constexpr( Model::synapse::size > 0 )
		for( std::size_t i = 0; i < net.num_synapses(); i++ )
			map( _synapses, net.get_synapse( i ), [i]( auto & dptr, auto elem ) {
				dptr[i] = elem;
			} );

	set();
	cudaDeviceSynchronize();
}


template <typename Model>
void snn<Model>::init( std::size_t num_neurons, float p_connect, float dt, int delay /* = 1 */ )
{
	init( neuron_group( num_neurons, p_connect ), dt, delay );
}
template <typename Model>
void snn<Model>::init( spice::util::neuron_group const & desc, float dt, int delay /* = 1 */ )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	spice::snn<Model>::init( dt, delay );

	static std::vector<spice::util::adj_list::int4> layout;
	auto const max_degree = adj_list::generate( desc, layout );

	reserve( desc.size(), max_degree, delay );

	static dev_ptr<spice::util::adj_list::int4> d_layout;
	d_layout.resize( layout.size() );
	cudaMemcpyAsync(
	    d_layout.data(), layout.data(), d_layout.size_in_bytes(), cudaMemcpyHostToDevice );

	generate_rnd_adj_list(
	    d_layout.data(),
	    narrow_int<int>( layout.size() ),
	    narrow_int<int>( desc.size() ),
	    narrow_int<int>( max_degree ),
	    _graph.edges.data() );
	_graph.edges.read_mostly();

	set();
	spice::cuda::init<Model>(
	    this->info(), {_graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() )} );
}


template <typename Model>
void snn<Model>::set()
{
	upload_storage<Model>( _neurons.data(), _synapses.data() );
}


template <typename Model>
adj_list const & snn<Model>::graph() const
{
	return _graph.adj;
}
template <typename Model>
typename Model::neuron::tuple_t snn<Model>::get_neuron( std::size_t const i ) const
{
	cudaDeviceSynchronize();
	return _neurons[i];
}
template <typename Model>
typename Model::synapse::tuple_t snn<Model>::get_synapse( std::size_t const i ) const
{
	cudaDeviceSynchronize();
	return _synapses[i];
}


template <typename Model>
void snn<Model>::_step( int const i, float const dt, std::vector<int> * out_spikes )
{
	zero_async( &_spikes.counts[circidx( i, this->delay() )] );

	if constexpr( Model::synapse::size > 0 )
		_spikes.num_updates.zero_async();

	update<Model>(
	    this->info(),
	    dt,
	    _spikes.ids.row( circidx( i, this->delay() ) ),
	    &_spikes.counts[circidx( i, this->delay() )],

	    _spikes.history,
	    _graph.ages.data(),
	    _spikes.updates.data(),
	    _spikes.num_updates.data(),
	    i,
	    this->delay(),
	    MAX_HISTORY(),
	    {_graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() )} );

	if( i >= this->delay() - 1 )
	{
		receive<Model>(
		    this->info(),
		    {_graph.edges.data(), narrow_int<int>( _graph.adj.max_degree() )},

		    _spikes.ids.row( circidx( i - this->delay() + 1, this->delay() ) ),
		    &_spikes.counts[circidx( i - this->delay() + 1, this->delay() )],

		    _graph.ages.data(),
		    _spikes.history,
		    MAX_HISTORY(),
		    i,
		    this->delay(),
		    dt );
	}

	if( out_spikes )
	{
		cudaDeviceSynchronize();
		out_spikes->assign(
		    _spikes.ids.row( circidx( i, this->delay() ) ),
		    _spikes.ids.row( circidx( i, this->delay() ) ) +
		        _spikes.counts[circidx( i, this->delay() )] );
	}
}


template class snn<spice::vogels_abbott>;
template class snn<spice::brunel>;
template class snn<spice::brunel_with_plasticity>;
template class snn<spice::synth>;
} // namespace spice::cuda
