#include <spice/cuda/multi_snn.h>

#include <spice/cuda/util/defs.h>
#include <spice/cuda/util/device.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>

#include <cuda_runtime.h>


using namespace spice::util;
using namespace spice::cuda::util;


namespace spice::cuda
{
template <typename Model>
multi_snn<Model>::multi_snn( float dt, int_ delay )
    : ::spice::snn<Model>( dt, delay )
    , _spikes{ { nullptr, cudaFreeHost } }
{
	void * ptr;
	success_or_throw(
	    cudaHostAlloc( &ptr, device::devices().size() * sizeof( uint_ ), cudaHostAllocPortable ) );

	_spikes.counts_data.reset( static_cast<uint_ *>( ptr ) );
	_spikes.counts = { _spikes.counts_data.get(), device::devices().size() };
}

template <typename Model>
multi_snn<Model>::multi_snn( spice::util::layout desc, float dt, int_ delay /* = 1 */ )
    : multi_snn<Model>( dt, delay )
{
	for( auto & d : device::devices() )
	{
		d.set();

		auto slice = desc.cut( device::devices().size(), d );
		_nets[d].emplace( slice.part, dt, delay, slice.first, slice.last );
	}
}

template <typename Model>
multi_snn<Model>::multi_snn( spice::snn<Model> const & net )
    : multi_snn<Model>( net.dt(), net.delay() )
{
	auto adj_data = net.adj();
	adj_list adj( adj_data.first.size() / adj_data.second, adj_data.second, adj_data.first.data() );

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
		_nets[d].emplace(
		    slice, deg, net.dt(), net.delay(), narrow<int>( first ), narrow<int>( last ) );
	}
}


template <typename Model>
void multi_snn<Model>::step( std::vector<int> * out_spikes /* = nullptr */ )
{
	int_ * spikes[device::max_devices];
	uint_ * n_spikes[device::max_devices];

	{
		std::vector<int> tmp;
		if( out_spikes ) out_spikes->clear();

		for( auto & d : device::devices() )
		{
			d.set();
			_nets[d]->step( &spikes[d], &n_spikes[d], out_spikes ? &tmp : nullptr );
			if( out_spikes ) out_spikes->insert( out_spikes->end(), tmp.begin(), tmp.end() );
		}
	}

	sync();

	// 2 gpus for now:
	success_or_throw( cudaMemcpy( &_spikes.counts[0], n_spikes[0], 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( &_spikes.counts[1], n_spikes[1], 4, cudaMemcpyDefault ) );

	success_or_throw( cudaMemcpy(
	    spikes[0] + _spikes.counts[0], spikes[1], 4 * _spikes.counts[1], cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy(
	    spikes[1] + _spikes.counts[1], spikes[0], 4 * _spikes.counts[0], cudaMemcpyDefault ) );

	_spikes.counts[0] += _spikes.counts[1];
	success_or_throw( cudaMemcpy( n_spikes[0], &_spikes.counts[0], 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( n_spikes[1], &_spikes.counts[0], 4, cudaMemcpyDefault ) );
}

template <typename Model>
void multi_snn<Model>::sync()
{
	for( auto & d : device::devices() ) d.synchronize();
}

template <typename Model>
std::size_t multi_snn<Model>::num_neurons() const
{
	return _nets.front().value().num_neurons();
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
	adj_data.reserve( device::devices().size() );

	for( std::size_t i = 0; i < device::devices().size(); i++ )
		adj_data.push_back( _nets[i]->adj() );

	std::vector<adj_list> adj;
	adj.reserve( device::devices().size() );

	for( std::size_t i = 0; i < device::devices().size(); i++ )
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
	for( std::size_t i = 0; i < device::devices().size(); i++ )
	{
		std::size_t const first = i * num_neurons() / device::devices().size();
		std::size_t const last = ( i + 1 ) * num_neurons() / device::devices().size();

		auto tmp = _nets[i]->neurons();
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