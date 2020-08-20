#include <spice/cuda/multi_snn.h>

#include <spice/cuda/util/defs.h>
#include <spice/cuda/util/device.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>


using namespace spice::util;
using namespace spice::cuda::util;


namespace spice::cuda
{
template <typename Model>
multi_snn<Model>::multi_snn( spice::util::layout desc, float dt, int delay /* = 1 */ )
    : ::spice::snn<Model>( dt, delay )
{
	_nets.reserve( device::devices().size() );
	for( auto & d : device::devices() )
	{
		d.set();

		auto slice = desc.cut( device::devices().size(), d );
		_nets.push_back( cuda::snn<Model>( slice.part, dt, delay, slice.first, slice.last ) );
	}
}

template <typename Model>
multi_snn<Model>::multi_snn( spice::snn<Model> const & net )
    : ::spice::snn<Model>( net )
{
	auto adj_data = net.adj();
	adj_list adj( adj_data.first.size() / adj_data.second, adj_data.second, adj_data.first.data() );

	_nets.reserve( device::devices().size() );
	std::vector<int> slice;
	for( auto & d : device::devices() )
	{
		// TODO: Load-balanced split
		std::size_t const first = d * net.num_neurons() / device::devices().size();
		std::size_t const last = ( d + 1 ) * net.num_neurons() / device::devices().size();

		std::size_t deg = 0;
		for( std::size_t i = 0; i < adj.num_nodes(); i++ )
		{
			auto row = adj.neighbors( i );
			deg = std::max(
			    deg,
			    static_cast<std::size_t>(
			        std::lower_bound( row.begin(), row.end(), last ) -
			        std::lower_bound( row.begin(), row.end(), first ) ) );
		}

		deg = ( deg + WARP_SZ - 1 ) / WARP_SZ * WARP_SZ;
		spice_assert( deg > 0 );
		slice.resize( deg * adj.num_nodes() );

		for( std::size_t i = 0; i < adj.num_nodes(); i++ )
		{
			auto row = adj.neighbors( i );
			auto const a = std::lower_bound( row.begin(), row.end(), first );
			auto const b = std::lower_bound( row.begin(), row.end(), last );

			std::copy( a, b, &slice.at( i * deg ) );
			std::fill( &slice.at( i * deg ) + ( b - a ), &slice.at( i * deg ) + deg, -1 );
		}

		d.set();
		_nets.push_back( cuda::snn<Model>(
		    slice, deg, net.dt(), net.delay(), narrow<int>( first ), narrow<int>( last ) ) );
	}
}


template <typename Model>
void multi_snn<Model>::step( std::vector<int> * out_spikes /* = nullptr */ )
{
	int * spikes[8];
	unsigned * n_spikes[8];

	{
		std::vector<int> tmp;
		if( out_spikes ) out_spikes->clear();

		for( auto & d : device::devices() )
		{
			d.set();
			_nets[d].step( &spikes[d], &n_spikes[d], out_spikes ? &tmp : nullptr );
			if( out_spikes ) out_spikes->insert( out_spikes->end(), tmp.begin(), tmp.end() );
		}
	}

	sync();

	// 2 gpus for now:
	int a, b;
	success_or_throw( cudaMemcpy( &a, n_spikes[0], 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( &b, n_spikes[1], 4, cudaMemcpyDefault ) );

	success_or_throw( cudaMemcpy( spikes[0] + a, spikes[1], 4 * b, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( spikes[1] + b, spikes[0], 4 * a, cudaMemcpyDefault ) );

	a += b;
	success_or_throw( cudaMemcpy( n_spikes[0], &a, 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( n_spikes[1], &a, 4, cudaMemcpyDefault ) );
}

template <typename Model>
void multi_snn<Model>::sync()
{
	for( auto & d : device::devices() ) d.synchronize();
}

template <typename Model>
std::size_t multi_snn<Model>::num_neurons() const
{
	return _nets.front().num_neurons();
}
template <typename Model>
std::size_t multi_snn<Model>::num_synapses() const
{
	return adj().first.size();
}

template <typename Model>
std::pair<std::vector<int>, std::size_t> multi_snn<Model>::adj() const
{
	// TODO: Load-balanced split
	std::vector<std::pair<std::vector<int>, std::size_t>> adj_data;
	adj_data.reserve( _nets.size() );

	for( std::size_t i = 0; i < _nets.size(); i++ ) adj_data.push_back( _nets[i].adj() );

	std::vector<adj_list> adj;
	adj.reserve( _nets.size() );

	for( std::size_t i = 0; i < _nets.size(); i++ )
		adj.push_back(
		    { adj_data[i].first.size() / adj_data[i].second,
		      adj_data[i].second,
		      adj_data[i].first.data() } );

	std::size_t deg = 0;
	for( std::size_t i = 0; i < num_neurons(); i++ )
	{
		std::size_t sum = 0;
		for( auto & a : adj ) sum += a.neighbors( i ).size();
		deg = std::max( deg, sum );
	}

	deg = ( deg + WARP_SZ - 1 ) / WARP_SZ * WARP_SZ;
	std::vector<int> result( deg * num_neurons() );

	for( std::size_t i = 0; i < num_neurons(); i++ )
	{
		std::size_t offset = 0;
		for( auto & a : adj )
		{
			auto row = a.neighbors( i );
			std::copy( row.begin(), row.end(), &result.at( i * deg ) + offset );
			offset += row.size();
		}
		std::fill( &result.at( i * deg ) + offset, &result.at( i * deg ) + deg, -1 );
	}

	return { result, deg };
}
template <typename Model>
std::vector<typename Model::neuron::tuple_t> multi_snn<Model>::neurons() const
{
	// TODO: Load-balanced split
	std::vector<typename Model::neuron::tuple_t> result;
	for( std::size_t i = 0; i < _nets.size(); i++ )
	{
		std::size_t const first = i * num_neurons() / device::devices().size();
		std::size_t const last = ( i + 1 ) * num_neurons() / device::devices().size();

		auto tmp = _nets[i].neurons();
		result.insert( result.end(), tmp.begin() + first, tmp.begin() + last );
	}
	return result;
}
template <typename Model>
std::vector<typename Model::synapse::tuple_t> multi_snn<Model>::synapses() const
{
	// TODO: Fix
	return {};
}

template class multi_snn<spice::vogels_abbott>;
template class multi_snn<spice::brunel>;
template class multi_snn<spice::brunel_with_plasticity>;
template class multi_snn<spice::synth>;
} // namespace spice::cuda