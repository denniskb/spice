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


static unsigned long long seed = 1337;


using namespace spice::util;


template <typename T, bool Const = false>
class iter
{
public:
	iter( T * data, unsigned i )
	    : _data( data )
	    , _i( i )
	{
	}

	unsigned id() const { return _i; }

#pragma warning( push )
#pragma warning( disable : 6011 ) // deref. null ptr - asserted not null
	template <int I, bool C = Const>
	auto const & get( typename std::enable_if_t<C> * dummy = 0 )
	{
		spice_assert( _data );
		return std::get<I>( _data[_i] );
	}

	template <int I, bool C = Const>
	auto & get( typename std::enable_if_t<!C> * dummy = 0 )
	{
		spice_assert( _data );
		return std::get<I>( _data[_i] );
	}
#pragma warning( pop )

private:
	T * _data = nullptr;
	unsigned _i = 0;
};

template <typename T>
using const_iter = iter<T, true>;


template <typename F, typename ID>
static void for_each( F && f, int const count, ID && id, adj_list const & adj )
{
	for( int i = 0; i < count; i++ )
	{
		int const src = std::forward<ID>( id )( i );

		int j = 0;
		for( int dst : adj.neighbors( src ) )
			std::forward<F>( f )( narrow<unsigned>( adj.edge_index( src, j++ ) ), src, dst );
	}
}


namespace spice::cpu
{
template <typename Model>
snn<Model>::snn( layout const & desc, float const dt, int const delay /* = 1 */ )
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
		for( int i = 0; i < desc.size(); i++ )
			Model::neuron::template init( iter( _neurons->data(), i ), this->info(), _backend );
	}

	// Init synapses
	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.emplace( _graph.adj.num_edges() );
		for_each(
		    [&]( unsigned syn, int src, int dst ) {
			    Model::synapse::template init(
			        iter( _synapses->data(), syn ), src, dst, this->info(), _backend );
		    },
		    narrow<int>( desc.size() ),
		    []( int x ) { return x; },
		    _graph.adj );

		_spikes.flags.emplace( delay + 1 );
		for( auto & bitvec : *_spikes.flags ) bitvec.resize( desc.size() );
	}
}

template <typename Model>
void snn<Model>::step( std::vector<int> * out_spikes )
{
	this->_step( [&]( int const istep, float const dt ) {
		int const post = istep % ( this->delay() + 1 );
		int const pre = ( istep + 1 ) % ( this->delay() + 1 );

		// Receive spikes
		if( _spikes.counts.size() >= this->delay() )
		{
			for_each(
			    [&]( int syn, int src, int dst ) {
				    Model::neuron::template receive(
				        src,
				        iter( _neurons->data(), dst ),
				        const_iter<Model::synapse::tuple_t>( _synapses->data(), syn ),
				        this->info(),
				        _backend );
			    },
			    narrow<int>( _spikes.counts.front() ),
			    [&]( int x ) { return _spikes.ids[x]; },
			    _graph.adj );

			_spikes.ids.erase( _spikes.ids.begin(), _spikes.ids.begin() + _spikes.counts.front() );
			_spikes.counts.erase( _spikes.counts.begin() );
		}

		// Update neurons
		{
			auto const nspikes = _spikes.ids.size();

			for( int i = 0; i < this->num_neurons(); i++ )
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
			    [&]( int syn, int src, int dst ) {
				    Model::synapse::template update(
				        iter( _synapses->data(), syn ),
				        src,
				        dst,
				        ( *_spikes.flags )[pre][src],
				        ( *_spikes.flags )[post][dst],
				        dt,
				        this->info(),
				        _backend );
			    },
			    narrow<int>( this->num_neurons() ),
			    []( int x ) { return x; },
			    _graph.adj );
		}
	} );
}


// TODO: Remove code duplication
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
std::pair<std::vector<int>, std::size_t> snn<Model>::adj() const
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


#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced formal paramter (VS doesn't recognize access to
                                  // 'istep' inside lambda)

#pragma warning( pop )


template class snn<vogels_abbott>;
template class snn<brunel>;
template class snn<brunel_with_plasticity>;
template class snn<synth>;
} // namespace spice::cpu
