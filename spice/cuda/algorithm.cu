#include "algorithm.h"

#include <spice/cuda/backend.cuh>
#include <spice/cuda/util/device.h>
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
#include <spice/util/stdint.h>

#include <array>


using namespace spice;
using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


__constant__ int4 _desc_range[20];
__constant__ float _desc_p[20];

__constant__ void * _neuron_storage[20];
__constant__ void * _synapse_storage[20];


static ulong_ seed()
{
	static ulong_ x = 1337;
	return hash( x++ );
}

static int_ nblocks( int_ desired, int_ max, int_ block_size )
{
	return std::min( max, ( desired + block_size - 1 ) / block_size );
}

class iter_base
{
public:
	__device__ explicit iter_base( int_ i )
	    : _i( i )
	{
	}

	__device__ uint_ id() const { return _i; }

private:
	uint_ const _i = 0;
};

template <typename Decl, bool Neuron = true, bool Const = false>
class iter : public iter_base
{
public:
	using iter_base::iter_base;

	template <int_ I, bool N = Neuron, bool C = Const>
	__device__ auto & get( typename std::enable_if_t<N && !C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _neuron_storage[I] )[id()];
	}

	template <int_ I, bool N = Neuron, bool C = Const>
	__device__ auto const & get( typename std::enable_if_t<N && C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _neuron_storage[I] )[id()];
	}

	template <int_ I, bool N = Neuron, bool C = Const>
	__device__ auto & get( typename std::enable_if_t<!N && !C> * dummy = 0 ) const
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Decl::tuple_t> *>(
		    _synapse_storage[I] )[id()];
	}

	template <int_ I, bool N = Neuron, bool C = Const>
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
    ulong_ const seed,
    int_ const desc_len,
    int_ const N,
    uint_ const max_degree,
    int_ * const out_edges )
{
	__shared__ float row[768];

	spice::util::xoroshiro128p rng( threadid() ^ seed );

	uint_ offset = blockIdx.x * max_degree;
	int_ total_degree = 0;
	for( int_ c = 0; c < desc_len; c++ )
	{
		if( blockIdx.x < _desc_range[c].x || blockIdx.x >= _desc_range[c].y ) continue;

		int_ first = _desc_range[c].z;
		int_ range = _desc_range[c].w - first;
		int_ degree = min( max_degree - total_degree, binornd( rng, range, _desc_p[c] ) );
		degree = __shfl_sync( MASK_ALL, degree, 0 );
		total_degree += degree;

		while( degree > 0 )
		{
			int_ const d = min( 768, degree );
			int_ const r = (int)( (long_)d * range / degree );

			// accumulate
			float total = 0.0f;
			for( int_ i = threadIdx.x; i < d; i += WARP_SZ )
			{
				float f = exprnd( rng );

				float sum;
				f = total + warp::inclusive_scan( f, sum, active_mask( i, d ) );
				total += sum;

				row[i] = f;
			}

			// normalize
			{
				total += exprnd( rng );
				total = __shfl_sync( MASK_ALL, total, 0 );

				float const scale = ( r - d ) / total;
				for( int_ i = threadIdx.x; i < d; i += WARP_SZ )
					( out_edges + offset )[i] = first + static_cast<int>( row[i] * scale ) + i;
			}

			offset += d;
			degree -= 768;
			first += r;
			range -= r;
		}
	}

	for( uint_ i = offset + threadIdx.x; i < ( blockIdx.x + 1 ) * max_degree; i += WARP_SZ )
		out_edges[i] = -1;
}

template <typename Model, bool INIT>
static __global__ void _process_neurons(
    int_ const first,
    int_ const last,
    snn_info const info,
    ulong_ const seed,

    float const dt = 0,
    int_ * spikes = nullptr,
    uint_ * num_spikes = nullptr,

    uint_ * history = nullptr,
    int_ const * ages = nullptr,
    int_ * updates = nullptr,
    uint_ * num_updates = nullptr,
    int_ const iter = 0,
    int_ const delay = 0,
    int_ const max_history = 0 )
{
	spice_assert( info.num_neurons < INT_MAX - num_threads() );

	backend bak( threadid() ^ seed );

	for( int_ i = first + threadid(); i < last; i += num_threads() )
	{
		neuron_iter<typename Model::neuron> it( i );

		if constexpr( INIT )
			Model::neuron::template init( it, info, bak );
		else // udpate
		{
			bool const spiked = Model::neuron::template update( it, dt, info, bak );

			if constexpr( Model::synapse::size > 0 ) // plast.
			{
				uint_ const flag = __ballot_sync( active_mask( i, last ), spiked );
				if( laneid() == 0 ) history[i / 32] = flag;

				if( iter - ages[i] + 1 == max_history )
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
    ulong_ const seed,
    span2d<int_ const> adj,

    int_ const * spikes = nullptr,
    uint_ const * num_spikes = nullptr,

    int_ * ages = nullptr,
    span2d<uint_> history = {},
    int_ const max_history = 0,
    int_ const iter = 0,
    int_ const delay = 0,
    float const dt = 0 )
{
	backend bak( threadid() ^ seed );

	for( int_ i = blockIdx.x; i < ( ( MODE == INIT_SYNS ) ? info.num_neurons : *num_spikes );
	     i += gridDim.x )
	{
		uint_ const src = ( MODE == INIT_SYNS ) ? i : spikes[i];

		for( uint_ j = threadIdx.x; j < adj.width(); j += blockDim.x )
		{
			uint_ const isyn = adj.row( src ) - adj.row( 0 ) + j;
			int_ const dst = adj( src, j );

			if( dst >= 0 )
			{
				if constexpr( MODE == INIT_SYNS )
					Model::synapse::template init(
					    synapse_iter<typename Model::synapse>( isyn ), src, dst, info, bak );
				else if constexpr( Model::synapse::size > 0 )
					for( int_ k = ages[src]; k <= iter; k++ )
						Model::synapse::template update(
						    synapse_iter<typename Model::synapse>( isyn ),
						    src,
						    dst,
						    history( circidx( k - delay, max_history ), src / 32 ) >> ( src % 32 ) &
						        1u,
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
    ulong_ const seed,
    span2d<int_ const> adj,

    int_ const * spikes = nullptr,
    uint_ const * num_spikes = nullptr )
{
	backend bak( threadid() ^ seed );

	int_ const S = *num_spikes;

	for( int_ i = blockIdx.x; i < S * ( adj.width() / WARP_SZ ); i += gridDim.x )
	{
		int_ s = i % S;
		int_ o = i / S;

		int_ src = spikes[s];
		int_ dst = adj( src, WARP_SZ * o + threadIdx.x );

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
void generate_rnd_adj_list( spice::util::layout const & desc, int_ * edges )
{
	spice_assert(
	    desc.connections().size() <= 20,
	    "spice doesn't support models with more than 20 connections between neuron populations" );
	spice_assert( edges || desc.size() * desc.max_degree() == 0 );

	std::array<int4, 20> tmp_range;
	std::array<float, 20> tmp_p;
	for( size_ i = 0; i < desc.connections().size(); i++ )
	{
		tmp_range[i].x = std::get<0>( desc.connections().at( i ) );
		tmp_range[i].y = std::get<1>( desc.connections().at( i ) );
		tmp_range[i].z = std::get<2>( desc.connections().at( i ) );
		tmp_range[i].w = std::get<3>( desc.connections().at( i ) );

		tmp_p[i] = std::get<4>( desc.connections().at( i ) );
	}

	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_range, tmp_range.data(), sizeof( int4 ) * desc.connections().size() ) );
	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_p, tmp_p.data(), sizeof( float ) * desc.connections().size() ) );

	cudaFuncSetCacheConfig( _generate_adj_ids, cudaFuncCachePreferShared );

	spice_assert( desc.size() <= ( 1u << 31 ) - 1 );
	call( [&] {
		_generate_adj_ids<<<narrow<int>( desc.size() ), WARP_SZ>>>(
		    seed(),
		    narrow<int>( desc.connections().size() ),
		    narrow<int>( desc.size() ),
		    narrow<uint_>( desc.max_degree() ),
		    edges );
	} );
}

template <typename Model>
void upload_meta(
    typename Model::neuron::ptuple_t const & neuron,
    typename Model::synapse::ptuple_t const & synapse )
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
void init( int_ first, int_ last, snn_info const info, span2d<int_ const> adj /* = {} */ )
{
	call( [&] {
		_process_neurons<Model, true>
		    <<<nblocks( last - first, 128, 256 ), 256>>>( first, last, info, seed() );
	} );

	if constexpr( Model::synapse::size > 0 )
		call( [&] { _process_spikes<Model, INIT_SYNS><<<128, 256>>>( info, seed(), adj ); } );
}
template void init<::spice::vogels_abbott>( int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::brunel>( int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::brunel_with_plasticity>( int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::synth>( int_, int_, snn_info, span2d<int_ const> );

template <typename Model>
void update(
    cudaStream_t s,
    cudaEvent_t updt,

    int_ first,
    int_ last,
    snn_info const info,
    float const dt,
    int_ * spikes,
    uint_ * num_spikes,

    span2d<uint_> history /* = {} */,
    int_ * ages /* = nullptr */,
    int_ * updates /* = nullptr */,
    uint_ * num_updates /* = nullptr */,
    int_ const iter /* = 0 */,
    int_ const delay /* = 0 */,
    int_ const max_history /* = 0 */,
    span2d<int_ const> adj /* = {} */ )
{
	call( [&] {
		_process_neurons<Model, false><<<128, 128, 0, s>>>(
		    first,
		    last,
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
		    max_history );
	} );

	if( updt ) success_or_throw( cudaEventRecord( updt, s ) );

	if constexpr( Model::synapse::size > 0 )
		call( [&] {
			_process_spikes<Model, UPDT_SYNS><<<256, 256, 0, s>>>(
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
    cudaStream_t,
    cudaEvent_t,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    span2d<uint_>,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const,
    int_ const,
    span2d<int_ const> );
template void update<::spice::brunel>(
    cudaStream_t,
    cudaEvent_t,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    span2d<uint_>,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const,
    int_ const,
    span2d<int_ const> );
template void update<::spice::brunel_with_plasticity>(
    cudaStream_t,
    cudaEvent_t,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    span2d<uint_>,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const,
    int_ const,
    span2d<int_ const> );
template void update<::spice::synth>(
    cudaStream_t,
    cudaEvent_t,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    span2d<uint_>,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const,
    int_ const,
    span2d<int_ const> );

template <typename Model>
void receive(
    cudaStream_t s,

    snn_info const info,
    span2d<int_ const> adj,

    int_ const * spikes,
    uint_ const * num_spikes,

    int_ * ages /* = nullptr */,
    span2d<uint_> history /* = {} */,
    int_ const max_history /* = 0 */,
    int_ const iter /* = 0 */,
    int_ const delay /* = 0 */,
    float const dt /* = 0 */ )
{
	// if( info.num_neurons < 400'000 * device::devices().size() || Model::synapse::size > 0 )
	call( [&] {
		_process_spikes<Model, HNDL_SPKS><<<256, 256, 0, s>>>(
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
	/*else
	    call( [&] {
	        _process_spikes_cache_aware<Model><<<512, WARP_SZ, 0, s>>>(
	            info,
	            seed(),
	            adj,

	            spikes,
	            num_spikes );
	    } );*/
}
template void receive<::spice::vogels_abbott>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ *,
    span2d<uint_>,
    int_ const,
    int_ const iter,
    int_ const delay,
    float const dt );
template void receive<::spice::brunel>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ *,
    span2d<uint_>,
    int_ const,
    int_ const iter,
    int_ const delay,
    float const dt );
template void receive<::spice::brunel_with_plasticity>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ *,
    span2d<uint_>,
    int_ const,
    int_ const iter,
    int_ const delay,
    float const dt );
template void receive<::spice::synth>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ *,
    span2d<uint_>,
    int_ const,
    int_ const iter,
    int_ const delay,
    float const dt );

template <typename T>
void zero_async( T * t, cudaStream_t s /* = nullptr */ )
{
	call( [&] { _zero_async<T><<<1, 1, 0, s>>>( t ); } );
}
template void zero_async<int>( int_ *, cudaStream_t );
template void zero_async<int64_t>( int64_t *, cudaStream_t );
template void zero_async<uint_>( uint_ *, cudaStream_t );
template void zero_async<uint64_t>( uint64_t *, cudaStream_t );
template void zero_async<float>( float *, cudaStream_t );
template void zero_async<double>( double *, cudaStream_t );
} // namespace cuda
} // namespace spice