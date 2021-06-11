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
#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced formal parameter 'num_synapses' for certain
                                  // template inst.
template <typename Model>
void snn<Model>::reserve( size_ const num_neurons, size_ const num_synapses, int_ const delay )
{
	spice_assert( num_synapses % num_neurons == 0 );

	_graph.edges.resize( num_synapses );
	_graph.adj = { num_neurons, num_synapses / num_neurons, _graph.edges.data() };
	_graph.pivots.resize( num_neurons * ( ( num_neurons + 1023 ) / 1024 ) );

	_spikes.ids_data.resize( 2 * delay * num_neurons );
	_spikes.ids = { _spikes.ids_data.data(), narrow<int>( num_neurons ) };
	_spikes.counts.resize( 2 * delay );
	_spikes.counts.zero_async( _sim );

	if constexpr( Model::neuron::size > 0 ) _neurons.resize( num_neurons );

	if constexpr( Model::synapse::size > 0 )
	{
		_synapses.resize( num_synapses );
		_spikes.history.resize( num_neurons );
		_spikes.updates.resize( num_neurons );
		_graph.ages.resize( num_neurons );

		_spikes.history.zero_async( _sim );
		_graph.ages.zero_async( _sim );
	}
}
#pragma warning( pop )


template <typename Model>
snn<Model>::snn(
    spice::util::layout const & desc,
    float const dt,
    int_ const delay /* = 1 */,
    int_ slice_width /* = -1 */,
    int_ n /* = 1 */,
    int_ i /* = 0 */ )
    : ::spice::snn<Model>( dt, delay )
    , _slice_width( slice_width == -1 ? narrow<int>( desc.size() ) : slice_width )
    , _n( n )
    , _i( i )
{
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	reserve( desc.size(), desc.size() * desc.max_degree(), delay );
	generate_adj_list( _sim, desc, _graph.edges.data() );
	generate_pivots(
	    _sim, desc.size(), desc.max_degree(), _graph.edges.data(), _graph.pivots.data() );

	upload_meta<Model>( _sim, _neurons.data(), _synapses.data() );
	spice::cuda::init<Model>(
	    _sim,
	    _slice_width,
	    _n,
	    _i,
	    this->info(),
	    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) } );
}

template <typename Model>
snn<Model>::snn(
    std::vector<int> const & adj,
    size_ width,
    float const dt,
    int_ const delay /* = 1 */,
    int_ slice_width /* = -1 */,
    int_ n /* = 1 */,
    int_ i /* = 0 */ )
    : ::spice::snn<Model>( dt, delay )
    , _slice_width( slice_width == -1 ? narrow<int>( adj.size() / width ) : slice_width )
    , _n( n )
    , _i( i )
{
	spice_assert( width > 0 );
	spice_assert( adj.size() % width == 0 );
	spice_assert( width % WARP_SZ == 0 );
	spice_assert( dt > 0.0f );
	spice_assert( delay >= 1 );

	reserve( adj.size() / width, adj.size(), delay );
	_graph.edges = adj;
	generate_pivots( _sim, adj.size() / width, width, _graph.edges.data(), _graph.pivots.data() );

	upload_meta<Model>( _sim, _neurons.data(), _synapses.data() );
	spice::cuda::init<Model>(
	    _sim,
	    _slice_width,
	    _n,
	    _i,
	    this->info(),
	    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) } );
}

template <typename Model>
snn<Model>::snn( spice::snn<Model> const & net )
    : ::spice::snn<Model>( net )
    , _slice_width( narrow<int>( net.num_neurons() ) )
    , _n( 1 )
    , _i( 0 )
{
	reserve( net.num_neurons(), net.num_synapses(), net.delay() );

	_graph.edges = net.adj().first;
	generate_pivots(
	    _sim,
	    net.num_neurons(),
	    net.num_synapses() / net.num_neurons(),
	    _graph.edges.data(),
	    _graph.pivots.data() );
	_neurons.from_aos( net.neurons() );
	_synapses.from_aos( net.synapses() );

	upload_meta<Model>( _sim, _neurons.data(), _synapses.data() );
}


template <typename Model>
void snn<Model>::step( std::vector<int> * out_spikes )
{
	int_ * spikes;
	uint_ * num_spikes;

	step( &spikes, &num_spikes, out_spikes );
}

template <typename Model>
void snn<Model>::step(
    int_ ** out_dspikes, uint_ ** out_dnum_spikes, std::vector<int> * out_spikes /* = nullptr */ )
{
	this->_step( [&]( int_ const i, float const dt ) {
		if constexpr( false || Model::synapse::size > 0 )
		{
			if( i >= this->delay() )
				receive<Model>(
				    _sim,

				    this->info(),
				    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) },

				    _spikes.ids.row( circidx( i, this->delay() ) ),
				    _spikes.counts.data() + circidx( i, this->delay() ),
				    _spikes.updates.data(),
				    _spikes.num_updates.data(),

				    _graph.ages.data(),
				    _spikes.history.data(),
				    i - 1,
				    dt );

			zero_async( _spikes.counts.data() + circidx( i, this->delay() ), _sim );
			if constexpr( Model::synapse::size > 0 ) _spikes.num_updates.zero_async( _sim );

			update<Model>(
			    _sim,

			    _slice_width,
			    _n,
			    _i,
			    this->info(),
			    dt,
			    _spikes.ids.row( circidx( i, this->delay() ) ),
			    _spikes.counts.data() + circidx( i, this->delay() ),

			    _spikes.history.data(),
			    _graph.ages.data(),
			    _spikes.updates.data(),
			    _spikes.num_updates.data(),
			    i,
			    this->delay() );

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
		}
		else
		{
			int_ const ping = i % 2 ? this->delay() : 0;
			int_ const pong = this->delay() - ping;

			zero_async( _spikes.counts.data() + pong, _sim, this->delay() );

			gather<Model>(
			    _sim,

			    this->info(),
			    dt,
			    this->delay(),
			    { _graph.edges.data(), narrow<int>( _graph.adj.max_degree() ) },
			    _graph.pivots.data(),

			    _spikes.ids.row( ping ),
			    _spikes.counts.data() + ping,

			    _spikes.ids.row( pong ),
			    _spikes.counts.data() + pong );

			if( out_spikes )
			{
				out_spikes->clear();

				for( int_ d = 0; d < this->delay(); d++ )
				{
					uint_ count;
					cudaMemcpy(
					    &count,
					    _spikes.counts.data() + pong + d,
					    sizeof( uint_ ),
					    cudaMemcpyDefault );

					auto const sz = out_spikes->size();
					out_spikes->resize( sz + count );
					cudaMemcpy(
					    out_spikes->data() + sz,
					    _spikes.ids.row( pong + d ),
					    count * sizeof( int_ ),
					    cudaMemcpyDefault );
				}
			}
		}
	} );
} // namespace spice::cuda

template <typename Model>
stream & snn<Model>::sim_stream()
{
	return _sim;
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