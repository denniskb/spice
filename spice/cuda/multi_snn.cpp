#include <spice/cuda/multi_snn.h>

#include <spice/cuda/util/defs.h>
#include <spice/cuda/util/device.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>

#include <cuda_runtime.h>

#include <immintrin.h>
#include <numeric>


using namespace spice::util;
using namespace spice::cuda::util;


static void copy( void * dst, void * src, size_ n, cudaStream_t s )
{
	success_or_throw( cudaMemcpyAsync( dst, src, n * 4, cudaMemcpyDefault, s ) );
};

static std::pair<int_, int_> batch( int_ const iter, int_ const delay )
{
	int_ const delta = std::max( 1, ( delay + static_cast<bool>( iter % delay ) ) / 2 );
	int_ const first = iter % delay;
	int_ const last = first + delta;

	return { first, last };
}


namespace spice::cuda
{
template <typename Model>
multi_snn<Model>::multi_snn(
    float dt, int_ delay, spice::util::layout const * desc /* = nullptr */ )
    : ::spice::snn<Model>( dt, delay )
    , _spikes{ { nullptr, cudaFreeHost } }
{
	void * ptr;
	success_or_throw( cudaHostAlloc(
	    &ptr, device::devices().size() * delay * sizeof( uint_ ), cudaHostAllocPortable ) );
	success_or_throw( cudaMemset( ptr, 0, device::devices().size() * delay * sizeof( uint_ ) ) );

	_spikes.counts_data.reset( static_cast<uint_ *>( ptr ) );
	_spikes.counts = { _spikes.counts_data.get(), delay };

	_spikes.ddata_data.resize( device::devices().size() * delay );
	_spikes.ddata = { _spikes.ddata_data.data(), delay };

	_spikes.dcounts_data.resize( device::devices().size() * delay, nullptr );
	_spikes.dcounts = { _spikes.dcounts_data.data(), delay };

	for( auto & d : device::devices() )
	{
		d.set();
		for( auto & d2 : device::devices() )
			if( d != d2 ) cudaDeviceEnablePeerAccess( d2, 0 );
	}
	// Absorb potential errors from blindly enabling peer access
	cudaGetLastError();

	_work += device::devices().size();

	for( auto & d : device::devices() )
		_workers[d] = std::thread(
		    [=]( int_ const ID ) {
			    device::devices( ID ).set();
			    _cp[ID].emplace();

			    if( desc )
			    {
				    auto slice = desc->cut( device::devices().size(), ID );
				    _nets[ID].emplace( slice.part, dt, delay, slice.first, slice.last );
			    }

			    sync_event updt;
			    auto const download_spikes = [&]( int_ const last ) {
				    auto [prev_first, prev_last] = batch( last, this->delay() );
				    _cp[ID]->wait( updt );
				    copy(
				        _spikes.counts.row( ID ) + prev_first,
				        *_spikes.dcounts.row( ID ) + prev_first,
				        prev_last - prev_first,
				        *_cp[ID] );
			    };

			    int iter = 0;
			    std::vector<int_> tmp;
			    _work--;
			    while( _running )
			    {
				    if( iter < _iter )
				    {
					    auto const [first, last] = batch( iter, this->delay() );

					    if( this->delay() > 1 && _spikes.dcounts( ID, last % this->delay() ) )
						    download_spikes( last );

					    for( int_ iter = first; iter < last; iter++ )
					    {
						    _nets[ID]->step(
						        &_spikes.ddata( ID, iter ),
						        &_spikes.dcounts( ID, iter ),
						        _out_spikes ? &tmp : nullptr );

						    if( _out_spikes )
						    {
							    std::lock_guard _( _out_spikes_lock );
							    _out_spikes->insert( _out_spikes->end(), tmp.begin(), tmp.end() );
						    }
					    }
					    updt.record( _nets[ID]->sim_stream() );

					    if( this->delay() == 1 ) download_spikes( last );
					    _cp[ID]->synchronize();
					    _work--;

					    iter += last - first;
				    }
				    std::this_thread::yield();
			    }
		    },
		    static_cast<int_>( d ) );

	while( _work ) std::this_thread::yield();
}

template <typename Model>
multi_snn<Model>::~multi_snn()
{
	_running = false;
	for( auto & d : device::devices() ) _workers[d].join();
}

template <typename Model>
multi_snn<Model>::multi_snn( spice::util::layout desc, float dt, int_ delay /* = 1 */ )
    : multi_snn<Model>( dt, delay, &desc )
{
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
	auto [first, last] = batch( _iter, this->delay() );
	bool const repeat = last < this->delay();

	_out_spikes = out_spikes;
	if( out_spikes && ( this->delay() == 1 || repeat ) ) out_spikes->clear();

	_work += device::devices().size();
	_iter += last - first;
	while( _work ) std::this_thread::yield(); // _mm_pause();

	std::tie( first, last ) = batch( last, this->delay() );

	if( device::devices().size() > 1 && [&] {
		    size_ total = 0;
		    for( auto & d : device::devices() )
			    total += std::accumulate(
			        _spikes.counts.row( d ) + first, _spikes.counts.row( d ) + last, 0 );
		    return total;
	    }() )
	{
		switch( device::devices().size() )
		{
		case 2:
			for( int_ step = first; step < last; step++ )
			{
				copy(
				    _spikes.ddata( 1, step ) + _spikes.counts( 1, step ),
				    _spikes.ddata( 0, step ),
				    _spikes.counts( 0, step ),
				    *_cp[0] );

				copy(
				    _spikes.ddata( 0, step ) + _spikes.counts( 0, step ),
				    _spikes.ddata( 1, step ),
				    _spikes.counts( 1, step ),
				    *_cp[1] );

				_spikes.counts( 0, step ) += _spikes.counts( 1, step );
			}
			break;

		case 4:
			for( int_ step = first; step < last; step++ )
			{
				copy(
				    _spikes.ddata( 1, step ) + _spikes.counts( 1, step ),
				    _spikes.ddata( 0, step ),
				    _spikes.counts( 0, step ),
				    *_cp[0] );

				copy(
				    _spikes.ddata( 3, step ) + _spikes.counts( 3, step ),
				    _spikes.ddata( 2, step ),
				    _spikes.counts( 2, step ),
				    *_cp[2] );

				_spikes.counts( 1, step ) += _spikes.counts( 0, step );
				_spikes.counts( 3, step ) += _spikes.counts( 2, step );
			}

			_cp[0]->synchronize();
			_cp[2]->synchronize();

			for( int_ step = first; step < last; step++ )
			{
				copy(
				    _spikes.ddata( 3, step ) + _spikes.counts( 3, step ),
				    _spikes.ddata( 1, step ),
				    _spikes.counts( 1, step ),
				    *_cp[1] );

				copy(
				    _spikes.ddata( 1, step ) + _spikes.counts( 1, step ),
				    _spikes.ddata( 3, step ),
				    _spikes.counts( 3, step ),
				    *_cp[3] );

				_spikes.counts( 3, step ) += _spikes.counts( 1, step );
				_spikes.counts( 1, step ) = _spikes.counts( 3, step );
			}

			_cp[1]->synchronize();
			_cp[3]->synchronize();

			for( int_ step = first; step < last; step++ )
			{
				copy(
				    _spikes.ddata( 0, step ),
				    _spikes.ddata( 1, step ),
				    _spikes.counts( 1, step ),
				    *_cp[1] );

				copy(
				    _spikes.ddata( 2, step ),
				    _spikes.ddata( 3, step ),
				    _spikes.counts( 3, step ),
				    *_cp[3] );

				_spikes.counts( 0, step ) = _spikes.counts( 1, step );
			}

			break;

		default:
			for( int_ step = first; step < last; step++ )
			{
				for( size_ i = 1; i < device::devices().size(); i++ )
				{
					copy(
					    _spikes.ddata( 0, step ) + _spikes.counts( 0, step ),
					    _spikes.ddata( i, step ),
					    _spikes.counts( i, step ),
					    *_cp[0] );
					_spikes.counts( 0, step ) += _spikes.counts( i, step );
				}

				for( size_ i = 1; i < device::devices().size(); i++ )
					copy(
					    _spikes.ddata( i, step ),
					    _spikes.ddata( 0, step ),
					    _spikes.counts( 0, step ),
					    *_cp[0] );
			}
		}

		for( auto & d : device::devices() )
			copy(
			    *_spikes.dcounts.row( d ) + first,
			    _spikes.counts.row( 0 ) + first,
			    last - first,
			    *_cp[d] );
		for( auto & d : device::devices() ) _cp[d]->synchronize();
	}

	if( repeat ) step( out_spikes );
} // namespace spice::cuda

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
