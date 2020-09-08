#include <spice/cuda/multi_snn.h>

#include <spice/cuda/util/defs.h>
#include <spice/cuda/util/device.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>

#include <cuda_runtime.h>

#include <immintrin.h>


using namespace spice::util;
using namespace spice::cuda::util;


static void copy( void * dst, void * src, size_ n, cudaStream_t s )
{
	success_or_throw( cudaMemcpyAsync( dst, src, n * 4, cudaMemcpyDefault, s ) );
};


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

	for( auto & d : device::devices() )
	{
		d.set();
		_cp[d].emplace();
		_updt[d].emplace();

		for( auto & d2 : device::devices() )
			if( d != d2 ) cudaDeviceEnablePeerAccess( d2, 0 );
	}
	// Absorb potential errors from blindly enabling peer access
	cudaGetLastError();

	for( auto & d : device::devices() )
		_workers[d] = std::thread(
		    [&]( int_ const ID ) {
			    device::devices()[ID].set();

			    int iter = 0;
			    std::vector<int_> tmp;
			    while( _running )
			    {
				    if( iter < _iter )
				    {
					    _nets[ID]->step(
					        *_updt[ID],
					        &_spikes.ddata[ID],
					        &_spikes.dcounts[ID],
					        _out_spikes ? &tmp : nullptr );
					    _cp[ID]->wait( *_updt[ID] );
					    copy( &_spikes.counts[ID], _spikes.dcounts[ID], 1, *_cp[ID] );

					    if( _out_spikes )
					    {
						    std::lock_guard _( _out_spikes_lock );
						    _out_spikes->insert( _out_spikes->end(), tmp.begin(), tmp.end() );
						    tmp.clear();
					    }

					    _work--;
					    iter++;
				    }
				    else
					    std::this_thread::yield();
			    }
		    },
		    static_cast<int_>( d ) );
}

template <typename Model>
multi_snn<Model>::~multi_snn()
{
	_running = false;
	for( auto & d : device::devices() ) _workers[d].join();
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
	_out_spikes = out_spikes;
	if( out_spikes ) out_spikes->clear();

	_work += device::devices().size();
	_iter++;
	while( _work ) _mm_pause();

	if( device::devices().size() == 1 ) return;

	if( device::devices().size() == 2 )
	{
		_cp[0]->synchronize();
		_cp[1]->synchronize();

		copy( _spikes.ddata[0] + _spikes.counts[0], _spikes.ddata[1], _spikes.counts[1], *_cp[0] );
		copy( _spikes.ddata[1] + _spikes.counts[1], _spikes.ddata[0], _spikes.counts[0], *_cp[1] );

		uint_ a = _spikes.counts[0];
		uint_ b = _spikes.counts[1];
		_spikes.counts[0] += _spikes.counts[1];

		if( b ) copy( _spikes.dcounts[0], &_spikes.counts[0], 1, *_cp[0] );
		if( a ) copy( _spikes.dcounts[1], &_spikes.counts[0], 1, *_cp[1] );

		if( b ) _cp[0]->synchronize();
		if( a ) _cp[1]->synchronize();
	}
	else
	{
		_cp[0]->synchronize();
		for( size_ i = 1; i < device::devices().size(); i++ )
		{
			_cp[i]->synchronize();
			copy(
			    _spikes.ddata[0] + _spikes.counts[0],
			    _spikes.ddata[i],
			    _spikes.counts[i],
			    *_cp[0] );
			_spikes.counts[0] += _spikes.counts[i];
		}

		for( size_ i = 1; i < device::devices().size(); i++ )
			copy( _spikes.ddata[i], _spikes.ddata[0], _spikes.counts[0], *_cp[0] );

		if( _spikes.counts[0] )
		{
			for( auto & d : device::devices() )
				copy( _spikes.dcounts[d], &_spikes.counts[0], 1, *_cp[d] );
			for( auto & d : device::devices() ) _cp[d]->synchronize();
		}
	}
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
