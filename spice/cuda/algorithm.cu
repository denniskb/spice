#include "algorithm.h"

#include <spice/cuda/backend.cuh>
#include <spice/cuda/util/error.h>
#include <spice/cuda/util/utility.cuh>
#include <spice/cuda/util/warp.cuh>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/assert.h>
#include <spice/util/circular_buffer.h>
#include <spice/util/random.h>

#include <array>


using namespace spice;
using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


__constant__ int3 _desc_gendeg[20];
__constant__ float _desc_p[20];
__constant__ int2 _desc_genids[20];

__constant__ void * _neuron_storage[20];
__constant__ void * _synapse_storage[20];


static unsigned long long seed()
{
	static unsigned long long x = 1337;
	return hash( x++ );
}

static int nblocks( int desired, int max, int block_size )
{
	return std::min( max, ( desired + block_size - 1 ) / block_size );
}

class iter_base
{
public:
	__device__ explicit iter_base( int i )
	    : _i( i )
	{
	}

	__device__ unsigned id() const { return _i; }

private:
	unsigned const _i = 0;
};

template <typename Decl, bool Neuron = true, bool Const = false>
class iter : public iter_base
{
public:
	using iter_base::iter_base;

	template <int I, bool N = Neuron, bool C = Const>
	__device__ auto & get( typename std::enable_if_t<N && !C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _neuron_storage[I] )[id()];
	}

	template <int I, bool N = Neuron, bool C = Const>
	__device__ auto const & get( typename std::enable_if_t<N && C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _neuron_storage[I] )[id()];
	}

	template <int I, bool N = Neuron, bool C = Const>
	__device__ auto & get( typename std::enable_if_t<!N && !C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _synapse_storage[I] )[id()];
	}

	template <int I, bool N = Neuron, bool C = Const>
	__device__ auto const & get( typename std::enable_if_t<!N && C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _synapse_storage[I] )[id()];
	}
};

template <typename Decl>
using neuron_iter = iter<Decl, true, false>;

template <typename Decl>
using const_neuron_iter = iter<Decl, true, true>;

template <typename Decl>
using synapse_iter = iter<Decl, false, false>;

template <typename Decl>
using const_synapse_iter = iter<Decl, false, true>;


static __global__ void _generate_adj_ids(
    unsigned long long const seed,
    int const desc_len,
    int const N,
    unsigned const max_degree,
    int * const out_edges )
{
	__shared__ float row[768];

	spice::util::xoroshiro128p rng( threadid() ^ seed );

	unsigned offset = blockIdx.x * max_degree;
	int total_degree = 0;
	for( int c = 0; c < desc_len; c++ )
	{
		if( blockIdx.x < _desc_gendeg[c].x || blockIdx.x >= _desc_gendeg[c].y ) continue;

		int degree =
		    min( max_degree - total_degree, binornd( rng, _desc_gendeg[c].z, _desc_p[c] ) );
		degree = __shfl_sync( MASK_ALL, degree, 0 );
		total_degree += degree;
		int first = _desc_genids[c].x;
		int range = _desc_genids[c].y;

		while( degree > 0 )
		{
			int const d = min( 768, degree );
			int const r = (int)( (long long)d * range / degree );

			// accumulate
			float total = 0.0f;
			for( int i = threadIdx.x; i < d; i += WARP_SZ )
			{
				float f = exprnd( rng );

				float sum;
				f = total + warp::inclusive_scan( f, sum, __activemask() );
				total += sum;

				row[i] = f;
			}

			// normalize
			{
				total += exprnd( rng );
				total = __shfl_sync( MASK_ALL, total, 0 );

				float const scale = ( r - d ) / total;
				for( int i = threadIdx.x; i < d; i += WARP_SZ )
					( out_edges + offset )[i] = first + static_cast<int>( row[i] * scale ) + i;
			}

			offset += d;
			degree -= 768;
			first += r;
			range -= r;
		}
	}

	for( unsigned i = offset + threadIdx.x; i < ( blockIdx.x + 1 ) * max_degree; i += WARP_SZ )
		out_edges[i] = -1;
}

template <typename Model, bool INIT>
static __global__ void _process_neurons(
    snn_info const info,
    unsigned long long const seed,

    float const dt = 0,
    int * spikes = nullptr,
    unsigned * num_spikes = nullptr,

    unsigned * history = nullptr,
    int const * ages = nullptr,
    int * updates = nullptr,
    unsigned * num_updates = nullptr,
    int const iter = 0,
    int const delay = 0,
    int const max_history = 0,
    unsigned * delayed_history = nullptr )
{
	spice_assert( info.num_neurons < INT_MAX - num_threads() );

	backend bak( threadid() ^ seed );

	for( int i = threadid(); i < info.num_neurons; i += num_threads() )
	{
		neuron_iter<typename Model::neuron> it( i );

		if constexpr( INIT )
			Model::neuron::template init( it, info, bak );
		else // udpate
		{
			bool const spiked = Model::neuron::template update( it, dt, info, bak );

			if( Model::synapse::size > 0 ) // plast.
			{
				unsigned const flag = __ballot_sync( __activemask(), spiked );
				if( laneid() == 0 ) history[i / 32] = flag;

				bool const delayed_spike = delayed_history[i / 32] >> ( i % 32 ) & 1u;

				if( iter - ages[i] == max_history - 1 && !delayed_spike )
					updates[atomicInc( num_updates, info.num_neurons )] = i;
			}

			if( spiked ) spikes[atomicInc( num_spikes, info.num_neurons )] = i;
		}
	}
}

enum mode
{
	INIT_SYNS,
	UPDT_SYNS,
	HNDL_SPKS
};
template <typename Model, mode MODE>
static __global__ void _process_spikes(
    snn_info const info,
    unsigned long long const seed,
    span2d<int const> adj,

    int const * spikes = nullptr,
    unsigned const * num_spikes = nullptr,

    int * ages = nullptr,
    span2d<unsigned> history = {},
    int const max_history = 0,
    int const iter = 0,
    int const delay = 0,
    float const dt = 0 )
{
	backend bak( threadid() ^ seed );

	for( int i = blockIdx.x; i < ( ( MODE == INIT_SYNS ) ? info.num_neurons : *num_spikes );
	     i += gridDim.x )
	{
		unsigned const src = ( MODE == INIT_SYNS ) ? i : spikes[i];

		for( unsigned j = threadIdx.x; j < adj.width(); j += blockDim.x )
		{
			unsigned const isyn = adj.row( src ) - adj.row( 0 ) + j;
			int const dst = adj( src, j );

			if( dst >= 0 )
			{
				if constexpr( MODE == INIT_SYNS )
					Model::synapse::template init(
					    synapse_iter<typename Model::synapse>( isyn ), src, dst, info, bak );
				else if constexpr( Model::synapse::size > 0 )
					for( int k = ages[src]; k <= iter; k++ )
						Model::synapse::template update(
						    synapse_iter<typename Model::synapse>( isyn ),
						    src,
						    dst,
						    MODE == HNDL_SPKS && k == iter,
						    history( circidx( k, max_history ), dst / 32 ) >> ( dst % 32 ) & 1u,
						    dt,
						    info,
						    bak );

				if constexpr( MODE == HNDL_SPKS )
					Model::neuron::template receive(
					    src,
					    neuron_iter<typename Model::neuron>( dst ),
					    const_synapse_iter<typename Model::synapse>( isyn ),
					    info,
					    bak );
			}
		}

		if( MODE != INIT_SYNS && Model::synapse::size > 0 )
		{
			__syncthreads();

			if( threadIdx.x == 0 ) ages[src] = iter + 1;
		}
	}
}

template <typename Model>
static __global__ void _process_spikes_cache_aware(
    snn_info const info,
    unsigned long long const seed,
    span2d<int const> adj,

    int const * spikes = nullptr,
    unsigned const * num_spikes = nullptr )
{
	backend bak( threadid() ^ seed );

	int const S = *num_spikes;

	for( int i = blockIdx.x; i < S * ( adj.width() / WARP_SZ ); i += gridDim.x )
	{
		int s = i % S;
		int o = i / S;

		int src = spikes[s];
		int dst = adj( src, WARP_SZ * o + threadIdx.x );

		if( dst >= 0 )
			Model::neuron::template receive(
			    src,
			    neuron_iter<typename Model::neuron>( dst ),
			    const_synapse_iter<typename Model::synapse>( 0 ),
			    info,
			    bak );
	}
}

template <typename T>
static __global__ void _zero_async( T * t )
{
	*t = T( 0 );
}


namespace spice
{
namespace cuda
{
void generate_rnd_adj_list( spice::util::neuron_group const & desc, int * edges )
{
	spice_assert(
	    desc.connections().size() <= 20,
	    "spice doesn't support models with more than 20 connections between neuron populations" );
	spice_assert( edges );

	std::array<int3, 20> tmp_deg;
	std::array<float, 20> tmp_p;
	std::array<int2, 20> tmp_ids;
	for( std::size_t i = 0; i < desc.connections().size(); i++ )
	{
		tmp_deg[i].x = desc.first( std::get<0>( desc.connections().at( i ) ) );
		tmp_deg[i].y = desc.last( std::get<0>( desc.connections().at( i ) ) );
		tmp_deg[i].z = desc.size( std::get<1>( desc.connections().at( i ) ) );

		tmp_p[i] = std::get<2>( desc.connections().at( i ) );

		tmp_ids[i].x = desc.first( std::get<1>( desc.connections().at( i ) ) );
		tmp_ids[i].y = desc.range( std::get<1>( desc.connections().at( i ) ) );
	}

	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_gendeg, tmp_deg.data(), sizeof( int3 ) * desc.connections().size() ) );
	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_p, tmp_p.data(), sizeof( float ) * desc.connections().size() ) );
	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_genids, tmp_ids.data(), sizeof( int2 ) * desc.connections().size() ) );

	cudaFuncSetCacheConfig( _generate_adj_ids, cudaFuncCachePreferShared );

	spice_assert( desc.size() <= ( 1u << 31 ) - 1 );
	call( [&]() {
		_generate_adj_ids<<<desc.size(), 32>>>(
		    seed(), desc.connections().size(), desc.size(), desc.max_degree(), edges );
	} );
}

template <typename Model>
void upload_meta( Model::neuron::ptuple_t const & neuron, Model::synapse::ptuple_t const & synapse )
{
	static_assert(
	    Model::neuron::size <= 20,
	    "spice doesn't support models with more than 20 neuron attributes" );
	static_assert(
	    Model::synapse::size <= 20,
	    "spice doesn't support models with more than 20 synapse attributes" );

	if constexpr( Model::neuron::size > 0 )
	{
		std::array<void *, Model::neuron::size> tmp;
		spice::util::for_each_i( neuron, [&]( auto p, auto i ) { tmp[i] = p; } );

		success_or_throw(
		    cudaMemcpyToSymbolAsync( _neuron_storage, tmp.data(), sizeof( void * ) * tmp.size() ) );
	}

	if constexpr( Model::synapse::size > 0 )
	{
		std::array<void *, Model::synapse::size> tmp;
		spice::util::for_each_i( synapse, [&]( auto p, auto i ) { tmp[i] = p; } );

		success_or_throw( cudaMemcpyToSymbolAsync(
		    _synapse_storage, tmp.data(), sizeof( void * ) * tmp.size() ) );
	}
}
template void upload_meta<::spice::vogels_abbott>(
    ::spice::vogels_abbott::neuron::ptuple_t const &,
    ::spice::vogels_abbott::synapse::ptuple_t const & );
template void upload_meta<::spice::brunel>(
    ::spice::brunel::neuron::ptuple_t const &, ::spice::brunel::synapse::ptuple_t const & );
template void upload_meta<::spice::brunel_with_plasticity>(
    ::spice::brunel_with_plasticity::neuron::ptuple_t const &,
    ::spice::brunel_with_plasticity::synapse::ptuple_t const & );
template void upload_meta<::spice::synth>(
    ::spice::synth::neuron::ptuple_t const &, ::spice::synth::synapse::ptuple_t const & );

// TOOD: Fuse these two into one function using conditional compilation ('if constexpr')
template <typename Model>
void init( snn_info const info, span2d<int const> adj /* = {} */ )
{
	call( [&]() {
		_process_neurons<Model, true>
		    <<<nblocks( info.num_neurons, 128, 256 ), 256>>>( info, seed() );
	} );

	if constexpr( Model::synapse::size > 0 )
		call( [&]() { _process_spikes<Model, INIT_SYNS><<<128, 256>>>( info, seed(), adj ); } );
}
template void init<::spice::vogels_abbott>( snn_info, span2d<int const> );
template void init<::spice::brunel>( snn_info, span2d<int const> );
template void init<::spice::brunel_with_plasticity>( snn_info, span2d<int const> );
template void init<::spice::synth>( snn_info, span2d<int const> );

template <typename Model>
void update(
    snn_info const info,
    float const dt,
    int * spikes,
    unsigned * num_spikes,

    span2d<unsigned> history /* = {} */,
    int * ages /* = nullptr */,
    int * updates /* = nullptr */,
    unsigned * num_updates /* = nullptr */,
    int const iter /* = 0 */,
    int const delay /* = 0 */,
    int const max_history /* = 0 */,
    span2d<int const> adj /* = {} */ )
{
	call( [&]() {
		_process_neurons<Model, false><<<nblocks( info.num_neurons, 128, 256 ), 256>>>(
		    info,
		    seed(),
		    dt,
		    spikes,
		    num_spikes,
		    history.row( circidx( iter, max_history ) ),
		    ages,
		    updates,
		    num_updates,
		    iter,
		    delay,
		    max_history,
		    history.row( circidx( iter - delay, max_history ) ) );
	} );

	if constexpr( Model::synapse::size > 0 )
		call( [&]() {
			_process_spikes<Model, UPDT_SYNS><<<256, 256>>>(
			    info,
			    seed(),
			    adj,

			    updates,
			    num_updates,

			    ages,
			    history,
			    max_history,
			    iter,
			    delay,
			    dt );
		} );
}
template void update<::spice::vogels_abbott>(
    snn_info,
    float,
    int *,
    unsigned *,
    span2d<unsigned>,
    int *,
    int *,
    unsigned *,
    int const,
    int const,
    int const,
    span2d<int const> );
template void update<::spice::brunel>(
    snn_info,
    float,
    int *,
    unsigned *,
    span2d<unsigned>,
    int *,
    int *,
    unsigned *,
    int const,
    int const,
    int const,
    span2d<int const> );
template void update<::spice::brunel_with_plasticity>(
    snn_info,
    float,
    int *,
    unsigned *,
    span2d<unsigned>,
    int *,
    int *,
    unsigned *,
    int const,
    int const,
    int const,
    span2d<int const> );
template void update<::spice::synth>(
    snn_info,
    float,
    int *,
    unsigned *,
    span2d<unsigned>,
    int *,
    int *,
    unsigned *,
    int const,
    int const,
    int const,
    span2d<int const> );

template <typename Model>
void receive(
    snn_info const info,
    span2d<int const> adj,

    int const * spikes,
    unsigned const * num_spikes,

    int * ages /* = nullptr */,
    span2d<unsigned> history /* = {} */,
    int const max_history /* = 0 */,
    int const iter /* = 0 */,
    int const delay /* = 0 */,
    float const dt /* = 0 */ )
{
	if( Model::synapse::size > 0 || info.num_neurons < 400'000 )
		call( [&]() {
			_process_spikes<Model, HNDL_SPKS>
			    <<<( Model::synapse::size > 0 ? 256 : 128 ),
			       ( info.num_neurons / adj.width() > 40 ? 128 : 256 )>>>(
			        info,
			        seed(),
			        adj,

			        spikes,
			        num_spikes,

			        ages,
			        history,
			        max_history,
			        iter,
			        delay,
			        dt );
		} );
	else
		call( [&]() {
			_process_spikes_cache_aware<Model><<<512, 32>>>(
			    info,
			    seed(),
			    adj,

			    spikes,
			    num_spikes );
		} );
}
template void receive<::spice::vogels_abbott>(
    snn_info const,
    span2d<int const>,

    int const *,
    unsigned const *,

    int *,
    span2d<unsigned>,
    int const,
    int const iter,
    int const delay,
    float const dt );
template void receive<::spice::brunel>(
    snn_info const,
    span2d<int const>,

    int const *,
    unsigned const *,

    int *,
    span2d<unsigned>,
    int const,
    int const iter,
    int const delay,
    float const dt );
template void receive<::spice::brunel_with_plasticity>(
    snn_info const,
    span2d<int const>,

    int const *,
    unsigned const *,

    int *,
    span2d<unsigned>,
    int const,
    int const iter,
    int const delay,
    float const dt );
template void receive<::spice::synth>(
    snn_info const,
    span2d<int const>,

    int const *,
    unsigned const *,

    int *,
    span2d<unsigned>,
    int const,
    int const iter,
    int const delay,
    float const dt );

template <typename T>
void zero_async( T * t, cudaStream_t s /* = nullptr */ )
{
	call( [&]() { _zero_async<T><<<1, 1, 0, s>>>( t ); } );
}
template void zero_async<int>( int *, cudaStream_t );
template void zero_async<int64_t>( int64_t *, cudaStream_t );
template void zero_async<unsigned>( unsigned *, cudaStream_t );
template void zero_async<uint64_t>( uint64_t *, cudaStream_t );
template void zero_async<float>( float *, cudaStream_t );
template void zero_async<double>( double *, cudaStream_t );
} // namespace cuda
} // namespace spice