#include "algorithm.h"

#include <spice/cuda/backend.cuh>
#include <spice/cuda/util/dbuffer.h>
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
#include <atomic>


using namespace spice;
using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


__constant__ int4 _desc_range[200];
__constant__ float _desc_p[200];

__constant__ void * _neuron_storage[20];
__constant__ void * _synapse_storage[20];


static ulong_ seed()
{
	static std::atomic_uint64_t x = 1337;
	return hash( x++ );
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
		out_edges[i] = INT_MAX;
}

__device__ int_ _lower_bound( int_ const * arr, int_ const len, int_ const x )
{
	int_ i = 0, j = len - 1;
	while( i < j )
	{
		int_ mid = ( i + j ) / 2;

		if( arr[mid] >= x )
			j = mid;
		else
			i = mid + 1;
	}

	return i + ( arr[i] < x );
}

static __global__ void _generate_pivots( int_ const * adj, int_ const deg, int_ * pivots )
{
	__shared__ uint_ bounds[1024];
	adj = adj + blockIdx.x * deg;
	int_ const pivot = threadIdx.x * 1024;
	uint_ const i = _lower_bound( adj, deg, pivot );
	bounds[threadIdx.x] = i;
	__syncthreads();
	pivots[blockIdx.x * blockDim.x + threadIdx.x] = ( i << 16 ) | bounds[threadIdx.x + 1];
}

template <typename Model, bool INIT>
static __global__ void _process_neurons(
    int_ const slice_width,
    int_ const n,
    int_ const igpu,
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
    int_ const delay = 0 )
{
	spice_assert( info.num_neurons < INT_MAX - num_threads() );
	spice_assert( n == 1 || slice_width % WARP_SZ == 0 );

	backend bak( threadid() ^ seed );

	for( int_ ii = threadid();; ii += num_threads() )
	{
		int_ const i = ( ii / slice_width * n + igpu ) * slice_width + ii % slice_width;
		if( i >= info.num_neurons ) return;

		neuron_iter<typename Model::neuron> it( i );

		if constexpr( INIT )
			Model::neuron::template init( it, info, bak );
		else // udpate
		{
			bool const spiked = Model::neuron::template update( it, dt, info, bak );

			if constexpr( Model::synapse::size > 0 ) // plast.
			{
				uint_ const hist = ( history[i] << 1 ) | spiked;
				history[i] = hist;

				if( !( ( hist >> ( delay - 1 ) ) & 1 ) && iter - ages[i] == 31 )
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
    uint_ * history = nullptr,
    int_ const iter = 0,
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

			if( dst < INT_MAX )
			{
				if constexpr( MODE == INIT_SYNS )
					Model::synapse::template init(
					    synapse_iter<typename Model::synapse>( isyn ), src, dst, info, bak );
				else if constexpr( Model::synapse::size > 0 )
					if( Model::synapse::plastic( src, dst, info ) )
					{
						int_ k = iter - ages[src];
						uint_ hist = history[dst] << ( 31 - k ) >> ( 31 - k );
						while( hist )
						{
							int_ const shift = 31 - __clz( hist );
							int_ const steps = k - shift + 1;
							k = shift;

							Model::synapse::template update(
							    synapse_iter<typename Model::synapse>( isyn ),
							    steps,
							    MODE == HNDL_SPKS && k == 0,
							    ( hist >> k ) & 1u,
							    dt,
							    info,
							    bak );

							hist = hist << ( 32 - k ) >> ( 32 - k );
							k--;
						}
						// TODO: fuse both
						if( k >= 0 )
							Model::synapse::template update(
							    synapse_iter<typename Model::synapse>( isyn ),
							    k + 1,
							    MODE == HNDL_SPKS,
							    false,
							    dt,
							    info,
							    bak );
					}

				/*if constexpr( MODE == HNDL_SPKS )
				    Model::neuron::template receive(
				        src,
				        neuron_iter<typename Model::neuron>( dst ),
				        const_synapse_iter<typename Model::synapse>( isyn ),
				        info,
				        bak );*/
			}
		}

		if constexpr( MODE != INIT_SYNS && Model::synapse::size > 0 )
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
		int_ const s = i % S;
		int_ const o = i / S;

		int_ const src = spikes[s];
		int_ const dst = adj( src, WARP_SZ * o + threadIdx.x );

		if( dst < INT_MAX )
			Model::neuron::template receive(
			    src,
			    neuron_iter<typename Model::neuron>( dst ),
			    const_synapse_iter<typename Model::synapse>( 0 ),
			    info,
			    bak );
	}
}

template <typename Neuron>
struct shared_iter
{
	int_ * const data;
	int_ const i;

	__device__ shared_iter( int_ * p, int_ id )
	    : data( p )
	    , i( id )
	{
	}

	template <int_ I>
	__device__ auto & get()
	{
		return reinterpret_cast<std::tuple_element_t<I, typename Neuron::tuple_t> &>(
		    data[I * 1024 + i] );
	}

	__device__ int_ id() const { return i; }
};

template <typename Model>
static __global__ void _gather(
    snn_info const info,
    ulong_ const seed,
    span2d<int_ const> adj,

    int_ const * spikes = nullptr,
    uint_ const * num_spikes = nullptr,

    int_ const * pivots = nullptr )
{
	int_ const I = threadid();
	if( I >= info.num_neurons ) return;

	int_ const deg_pivots = ( info.num_neurons + 1023 ) / 1024 + 1;
	backend bak( I ^ seed );

	__shared__ int state[Model::neuron::size * 1024];
	neuron_iter<typename Model::neuron> nit( I );
	shared_iter<typename Model::neuron> sit( state, threadIdx.x );

	// TODO: Generalize
	if constexpr( Model::neuron::size > 0 ) get<0>( sit ) = get<0>( nit );
	if constexpr( Model::neuron::size > 1 ) get<1>( sit ) = get<1>( nit );
	if constexpr( Model::neuron::size > 2 ) get<2>( sit ) = get<2>( nit );
	if constexpr( Model::neuron::size > 3 ) get<3>( sit ) = get<3>( nit );
	__syncthreads();

	for( int_ s = warpid_block(); s < *num_spikes; s += 32 )
	{
		int_ const src = spikes[s];
		uint_ const bounds = pivots[src * deg_pivots + blockIdx.x];
		int_ const first = bounds >> 16;
		int_ const last = bounds & 0xffff;

		for( int_ i = first + laneid(); i < last; i += 32 )
		{
			int_ const dst = adj( src, i ) % 1024;

			Model::neuron::template receive(
			    src,
			    shared_iter<typename Model::neuron>( state, dst ),
			    const_synapse_iter<typename Model::synapse>( i + src * adj.width() ),
			    info,
			    bak );
		}
	}
	__syncthreads();

	if constexpr( Model::neuron::size > 0 ) get<0>( nit ) = get<0>( sit );
	if constexpr( Model::neuron::size > 1 ) get<1>( nit ) = get<1>( sit );
	if constexpr( Model::neuron::size > 2 ) get<2>( nit ) = get<2>( sit );
	if constexpr( Model::neuron::size > 3 ) get<3>( nit ) = get<3>( sit );
}

template <typename T>
static __global__ void _zero_async( T * t )
{
	*t = T( 0 );
}


static spice::cuda::util::dbuffer<int_> pivots;

namespace spice
{
namespace cuda
{
void generate_rnd_adj_list( cudaStream_t s, spice::util::layout const & desc, int_ * edges )
{
	spice_assert(
	    desc.connections().size() <= 200,
	    "spice doesn't support models with more than 200 connections between neuron "
	    "populations" );
	spice_assert( edges || desc.size() * desc.max_degree() == 0 );

	std::array<int4, 200> tmp_range;
	std::array<float, 200> tmp_p;
	for( size_ i = 0; i < desc.connections().size(); i++ )
	{
		tmp_range[i].x = std::get<0>( desc.connections().at( i ) );
		tmp_range[i].y = std::get<1>( desc.connections().at( i ) );
		tmp_range[i].z = std::get<2>( desc.connections().at( i ) );
		tmp_range[i].w = std::get<3>( desc.connections().at( i ) );

		tmp_p[i] = std::get<4>( desc.connections().at( i ) );
	}

	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_range,
	    tmp_range.data(),
	    sizeof( int4 ) * desc.connections().size(),
	    0,
	    cudaMemcpyDefault,
	    s ) );
	success_or_throw( cudaMemcpyToSymbolAsync(
	    _desc_p,
	    tmp_p.data(),
	    sizeof( float ) * desc.connections().size(),
	    0,
	    cudaMemcpyDefault,
	    s ) );

	cudaFuncSetCacheConfig( _generate_adj_ids, cudaFuncCachePreferShared );

	spice_assert( desc.size() <= ( 1u << 31 ) - 1 );
	call( [&] {
		_generate_adj_ids<<<narrow<int>( desc.size() ), WARP_SZ, 0, s>>>(
		    seed(),
		    narrow<int>( desc.connections().size() ),
		    narrow<int>( desc.size() ),
		    narrow<uint_>( desc.max_degree() ),
		    edges );

		pivots.resize( desc.size() * ( ( desc.size() + 1023 ) / 1024 + 1 ) );
		generate_pivots( s, desc, edges, pivots.data() );
	} );
}

void generate_pivots(
    cudaStream_t s, spice::util::layout const & desc, int_ const * adj, int_ * pivots )
{
	_generate_pivots<<<desc.size(), ( desc.size() + 1023 ) / 1024 + 1, 0, s>>>(
	    adj, desc.max_degree(), pivots );
}

template <typename Model>
void upload_meta(
    cudaStream_t s,
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

		success_or_throw( cudaMemcpyToSymbolAsync(
		    _neuron_storage, tmp.data(), sizeof( void * ) * tmp.size(), 0, cudaMemcpyDefault, s ) );
	}

	if constexpr( Model::synapse::size > 0 )
	{
		std::array<void *, Model::synapse::size> tmp;
		spice::util::for_each_i( synapse, [&]( auto p, auto i ) { tmp[i] = p; } );

		success_or_throw( cudaMemcpyToSymbolAsync(
		    _synapse_storage,
		    tmp.data(),
		    sizeof( void * ) * tmp.size(),
		    0,
		    cudaMemcpyDefault,
		    s ) );
	}
}
template void upload_meta<::spice::vogels_abbott>(
    cudaStream_t,
    ::spice::vogels_abbott::neuron::ptuple_t const &,
    ::spice::vogels_abbott::synapse::ptuple_t const & );
template void upload_meta<::spice::brunel>(
    cudaStream_t,
    ::spice::brunel::neuron::ptuple_t const &,
    ::spice::brunel::synapse::ptuple_t const & );
template void upload_meta<::spice::brunel_with_plasticity>(
    cudaStream_t,
    ::spice::brunel_with_plasticity::neuron::ptuple_t const &,
    ::spice::brunel_with_plasticity::synapse::ptuple_t const & );
template void upload_meta<::spice::synth>(
    cudaStream_t,
    ::spice::synth::neuron::ptuple_t const &,
    ::spice::synth::synapse::ptuple_t const & );

// TOOD: Fuse these two into one function using conditional compilation ('if constexpr')
template <typename Model>
void init(
    cudaStream_t s,
    int_ slice_width,
    int_ n,
    int_ i,

    snn_info const info,
    span2d<int_ const> adj /* = {} */ )
{
	spice_assert( slice_width > 0 );
	spice_assert( i >= 0 );
	spice_assert( i < n );
	spice_assert( n == 1 || slice_width % WARP_SZ == 0, "slice_width must be a multiple of 32" );

	call( [&] {
		_process_neurons<Model, true><<<256, 256, 0, s>>>( slice_width, n, i, info, seed() );
	} );

	if constexpr( Model::synapse::size > 0 )
		call( [&] { _process_spikes<Model, INIT_SYNS><<<256, 256, 0, s>>>( info, seed(), adj ); } );
}
template void
    init<::spice::vogels_abbott>( cudaStream_t, int_, int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::brunel>( cudaStream_t, int_, int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::brunel_with_plasticity>(
    cudaStream_t, int_, int_, int_, snn_info, span2d<int_ const> );
template void init<::spice::synth>( cudaStream_t, int_, int_, int_, snn_info, span2d<int_ const> );

template <typename Model>
void update(
    cudaStream_t s,

    int_ slice_width,
    int_ n,
    int_ i,
    snn_info const info,
    float const dt,
    int_ * spikes,
    uint_ * num_spikes,

    uint_ * history /* = {} */,
    int_ * ages /* = nullptr */,
    int_ * updates /* = nullptr */,
    uint_ * num_updates /* = nullptr */,
    int_ const iter /* = 0 */,
    int_ const delay /* = 0 */ )
{
	spice_assert( slice_width > 0 );
	spice_assert( i >= 0 );
	spice_assert( i < n );
	spice_assert( n == 1 || slice_width % WARP_SZ == 0, "slice_width must be a multiple of 32" );

	call( [&] {
		_process_neurons<Model, false><<<256, 256, 0, s>>>(
		    slice_width,
		    n,
		    i,
		    info,
		    seed(),
		    dt,
		    spikes,
		    num_spikes,
		    history,
		    ages,
		    updates,
		    num_updates,
		    iter,
		    delay );
	} );
}
template void update<::spice::vogels_abbott>(
    cudaStream_t,
    int_,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    uint_ *,
    int_ *,
    int_ *,
    uint_ *,
    int_ constt,
    int_ const );
template void update<::spice::brunel>(
    cudaStream_t,
    int_,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    uint_ *,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const );
template void update<::spice::brunel_with_plasticity>(
    cudaStream_t,
    int_,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    uint_ *,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const );
template void update<::spice::synth>(
    cudaStream_t,
    int_,
    int_,
    int_,
    snn_info,
    float,
    int_ *,
    uint_ *,
    uint_ *,
    int_ *,
    int_ *,
    uint_ *,
    int_ const,
    int_ const );

template <typename Model>
void receive(
    cudaStream_t s,

    snn_info const info,
    span2d<int_ const> adj,

    int_ const * spikes,
    uint_ const * num_spikes,
    int_ const * updates,
    uint_ const * num_updates,

    int_ * ages /* = nullptr */,
    uint_ * history /* = nullptr */,
    int_ const iter /* = 0 */,
    float const dt /* = 0 */ )
{
	// TODO
	if( info.num_neurons < 800'000 || Model::synapse::size > 0 )
		call( [&] {
			//*
			int_ const nblocks = Model::synapse::size > 0 ? 256 : 512;
			_process_spikes<Model, HNDL_SPKS><<<nblocks, 65536 / nblocks, 0, s>>>(
			    info,
			    seed(),
			    adj,

			    spikes,
			    num_spikes,

			    ages,
			    history,
			    iter,
			    dt );
			//*/
			_gather<Model><<<( info.num_neurons + 1023 ) / 1024, 1024, 0, s>>>(
			    info, seed(), adj, spikes, num_spikes, pivots.data() );
			//*/
		} );
	else
		call( [&] {
			_process_spikes_cache_aware<Model><<<2048, WARP_SZ, 0, s>>>(
			    info,
			    seed(),
			    adj,

			    spikes,
			    num_spikes );
		} );

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
			    iter,
			    dt );
		} );
}
template void receive<::spice::vogels_abbott>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ const *,
    uint_ const *,
    int_ *,
    uint_ *,
    int_ const iter,
    float const dt );
template void receive<::spice::brunel>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ const *,
    uint_ const *,
    int_ *,
    uint_ *,
    int_ const iter,
    float const dt );
template void receive<::spice::brunel_with_plasticity>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ const *,
    uint_ const *,
    int_ *,
    uint_ *,
    int_ const iter,
    float const dt );
template void receive<::spice::synth>(
    cudaStream_t,
    snn_info const,
    span2d<int_ const>,
    int_ const *,
    uint_ const *,
    int_ const *,
    uint_ const *,
    int_ *,
    uint_ *,
    int_ const iter,
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