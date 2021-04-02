#include "snn.h"

#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/assert.h>
#include <spice/util/random.h>
#include <spice/util/type_traits.h>

#include <algorithm>
#include <future>
#include <numeric>


static ulong_ seed = 1337;


using namespace spice::util;


template <typename T, bool Const = false>
class iter
{
public:
	iter( T * data, size_ i )
	    : _data( data )
	    , _i( i )
	{
	}

	size_ id() const { return _i; }

	template <int_ I, bool C = Const>
	auto const & get( typename std::enable_if_t<C> * dummy = 0 )
	{
		spice_assert( _data );
		return std::get<I>( _data[_i] );
	}

	template <int_ I, bool C = Const>
	auto & get( typename std::enable_if_t<!C> * dummy = 0 )
	{
		spice_assert( _data );
		return std::get<I>( _data[_i] );
	}

private:
	T * _data = nullptr;
	size_ _i = 0;
};

template <typename T>
using const_iter = iter<T, true>;


template <typename F, typename ID>
static void for_each( F && f, int_ const count, ID && id, adj_list const & adj )
{
	for( int_ i = 0; i < count; i++ )
	{
		int_ const src = std::forward<ID>( id )( i );

		int_ j = 0;
		for( int_ dst : adj.neighbors( src ) )
			std::forward<F>( f )( narrow<uint_>( adj.edge_index( src, j++ ) ), src, dst );
	}
}


namespace spice::cpu
{
template <typename Model>
snn<Model>::snn( layout const & desc, float const dt, int_ const delay /* = 1 */ )
    : ::spice::snn<Model>( dt, delay )
    , _backend( seed++ )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	{
		adj_list::generate( desc, _graph.edges );
		_graph.adj = { desc.size(), desc.max_degree(), _graph.edges.data() };
	}

	// Init neurons
	if constexpr( Model::neuron::size > 0 )
	{
		_neurons.emplace( desc.size() );
		for( size_ i = 0; i < desc.size(); i++ )
			Model::neuron::template init( iter( _neurons->data(), i ), this->info(), _backend );
	}

	// Init synapses
	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.emplace( _graph.adj.num_edges() );
		for_each(
		    [&]( uint_ syn, int_ src, int_ dst ) {
			    Model::synapse::template init(
			        iter( _synapses->data(), syn ), src, dst, this->info(), _backend );
		    },
		    narrow<int>( desc.size() ),
		    []( int_ x ) { return x; },
		    _graph.adj );

		_spikes.flags.emplace( delay + 1 );
		for( auto & bitvec : *_spikes.flags ) bitvec.resize( desc.size() );
	}
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
template <typename Model>
void snn<Model>::step( std::vector<int> * out_spikes )
{
	this->_step( [&]( int_ const istep, float const dt ) {
		int_ const post = istep % ( this->delay() + 1 );
		int_ const pre = ( istep + 1 ) % ( this->delay() + 1 );

		// Receive spikes
		if( _spikes.counts.size() >= static_cast<uint_>( this->delay() ) )
		{
			for_each(
			    [&]( int_ syn, int_ src, int_ dst ) {
				    Model::neuron::template receive(
				        src,
				        iter( _neurons->data(), dst ),
				        const_iter<typename Model::synapse::tuple_t>( _synapses->data(), syn ),
				        this->info(),
				        _backend );
			    },
			    narrow<int>( _spikes.counts.front() ),
			    [&]( int_ x ) { return _spikes.ids[x]; },
			    _graph.adj );

			_spikes.ids.erase( _spikes.ids.begin(), _spikes.ids.begin() + _spikes.counts.front() );
			_spikes.counts.erase( _spikes.counts.begin() );
		}

		// Update neurons
		{
			auto const nspikes = _spikes.ids.size();

			for( int_ i = 0; i < narrow<int>( this->num_neurons() ); i++ )
			{
				bool const spiked = Model::neuron::template update(
				    iter( _neurons ? _neurons->data() : nullptr, i ), dt, this->info(), _backend );

				if constexpr( Model::synapse::size > 0 ) ( *( _spikes.flags ) )[post][i] = spiked;

				if( spiked ) _spikes.ids.push_back( i );
			}

			_spikes.counts.push_back( _spikes.ids.size() - nspikes );
		}

		if( out_spikes && !_spikes.counts.empty() )
			out_spikes->assign( _spikes.ids.end() - _spikes.counts.back(), _spikes.ids.end() );

		// Update synapses
		if constexpr( Model::synapse::size > 0 )
		{
			for_each(
			    [&]( int_ syn, int_ src, int_ dst ) {
				    if( Model::synapse::plastic( src, dst, this->info() ) )
					    Model::synapse::template update(
					        iter( _synapses->data(), syn ),
					        1,
					        ( *_spikes.flags )[pre][src],
					        ( *_spikes.flags )[post][dst],
					        dt,
					        this->info(),
					        _backend );
			    },
			    narrow<int>( this->num_neurons() ),
			    []( int_ x ) { return x; },
			    _graph.adj );
		}
	} );
}
#pragma GCC diagnostic pop


// TODO: Remove code duplication
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
	return _neurons.value_or( std::vector<typename Model::neuron::tuple_t>{} );
}
template <typename Model>
std::vector<typename Model::synapse::tuple_t> snn<Model>::synapses() const
{
	return _synapses.value_or( std::vector<typename Model::synapse::tuple_t>{} );
}


template class snn<vogels_abbott>;
template class snn<brunel>;
template class snn<brunel_with_plasticity>;
template class snn<synth>;
} // namespace spice::cpu
