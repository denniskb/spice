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
		size_ const first = d * net.num_neurons() / device::devices().size();
		size_ const last = ( d + 1 ) * net.num_neurons() / device::devices().size();

		size_ deg = 0;
		for( size_ i = 0; i < adj.num_nodes(); i++ )
		{
			auto row = adj.neighbors( i );
			deg = std::max(
			    deg,
			    static_cast<size_>(
			        std::lower_bound( row.begin(), row.end(), last ) -
			        std::lower_bound( row.begin(), row.end(), first ) ) );
		}

		deg = ( deg + WARP_SZ - 1 ) / WARP_SZ * WARP_SZ;
		spice_assert( deg > 0 );
		slice.resize( deg * adj.num_nodes() );

		for( size_ i = 0; i < adj.num_nodes(); i++ )
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

	auto copy = []( void * dst, void * src, size_ n ) {
		success_or_throw( cudaMemcpy( dst, src, n * 4, cudaMemcpyDefault ) );
	};

	// download spike counts
	for( auto & d : device::devices() ) copy( &_spikes.counts[d], n_spikes[d], 1 );

	// gather spike
	for( size_ delta = 1; delta < device::devices().size(); delta *= 2 )
	{
		for( size_ i = 0; i < device::devices().size() - delta; i += 2 * delta )
		{
			bool const last_iter = ( 2 * delta >= device::devices().size() );

			copy( spikes[i] + _spikes.counts[i], spikes[i + delta], _spikes.counts[i + delta] );
			if( last_iter )
				copy( spikes[i + delta] + _spikes.counts[i + delta], spikes[i], _spikes.counts[i] );

			_spikes.counts[i] += _spikes.counts[i + delta];
			if( last_iter ) _spikes.counts[i + delta] += _spikes.counts[i];
		}
		// sync
	}

	// scatter spikes
	for( size_ delta =
	         narrow_cast<size_>( std::log2( std::max( 1_sz, device::devices().size() - 1 ) ) );
	     delta >= 1;
	     delta /= 2 )
	{
		for( size_ i = 0; i < device::devices().size() - delta; i += 2 * delta )
			copy( spikes[i + delta], spikes[i], _spikes.counts[0] );

		// sync
	}

	// upload spike counts
	for( auto & d : device::devices() ) copy( n_spikes[d], &_spikes.counts[0], 1 );
}

template <typename Model>
void multi_snn<Model>::sync()
{
	for( auto & d : device::devices() ) d.synchronize();
}

template <typename Model>
size_ multi_snn<Model>::num_neurons() const
{
	return _nets.front().value().num_neurons();
}
template <typename Model>
size_ multi_snn<Model>::num_synapses() const
{
	return adj().first.size();
}

template <typename Model>
std::pair<std::vector<int>, size_> multi_snn<Model>::adj() const
{
	// TODO: Load-balanced split
	std::vector<std::pair<std::vector<int>, size_>> adj_data;
	adj_data.reserve( device::devices().size() );

	for( size_ i = 0; i < device::devices().size(); i++ ) adj_data.push_back( _nets[i]->adj() );

	std::vector<adj_list> adj;
	adj.reserve( device::devices().size() );

	for( size_ i = 0; i < device::devices().size(); i++ )
		adj.push_back(
		    { adj_data[i].first.size() / adj_data[i].second,
		      adj_data[i].second,
		      adj_data[i].first.data() } );

	size_ deg = 0;
	for( size_ i = 0; i < num_neurons(); i++ )
	{
		size_ sum = 0;
		for( auto & a : adj ) sum += a.neighbors( i ).size();
		deg = std::max( deg, sum );
	}

	deg = ( deg + WARP_SZ - 1 ) / WARP_SZ * WARP_SZ;
	std::vector<int> result( deg * num_neurons() );

	for( size_ i = 0; i < num_neurons(); i++ )
	{
		size_ offset = 0;
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
	for( size_ i = 0; i < device::devices().size(); i++ )
	{
		size_ const first = i * num_neurons() / device::devices().size();
		size_ const last = ( i + 1 ) * num_neurons() / device::devices().size();

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
