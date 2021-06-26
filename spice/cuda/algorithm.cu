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
	__device__ explicit iter_base( uint_ i )
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
    int_ const max_degree,
    __restrict__ int_ * out_edges )
{
	spice::util::xoroshiro128p rng( threadid() ^ seed );

	out_edges += blockIdx.x * max_degree;

	int_ total_degree = 0;
	for( int_ c = 0; c < desc_len; c++ )
	{
		if( blockIdx.x < _desc_range[c].x || blockIdx.x >= _desc_range[c].y ) continue;

		int_ const first = _desc_range[c].z;
		int_ const range = _desc_range[c].w - first;
		int_ const degree = __shfl_sync(
		    MASK_ALL, min( max_degree - total_degree, binornd( rng, range, _desc_p[c] ) ), 0 );
		int_ const tail = degree - floor32( degree );
		total_degree += degree;

		auto gen = [&, total = 0.0f, residue = __shfl_sync( MASK_ALL, exprnd( rng ), 0 )](
		               uint_ const mask, float const expected_sum, int_ const i ) mutable {
			float sum;
			float f = total + warp::inclusive_scan( exprnd( rng ), sum, mask );
			total += sum;
			residue += sum - expected_sum;

			out_edges[i] =
			    first + i +
			    static_cast<int_>( roundf( f * ( range - degree ) / ( degree + residue ) ) );
		};

		int_ i = threadIdx.x;
		for( ; i < floor32( degree ); i += WARP_SZ ) gen( MASK_ALL, WARP_SZ, i );
		if( i < degree ) gen( set_lsbs( tail ), tail, i );

		out_edges += degree;
	}

	for( int_ i = threadIdx.x; i < max_degree - total_degree; i += WARP_SZ ) out_edges[i] = INT_MAX;
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

static __global__ void
_generate_pivots( __restrict__ int_ const * adj, int_ const deg, __restrict__ uint_ * pivots )
{
	__shared__ uint_ bounds[1024];

	adj = adj + blockIdx.x * deg;
	int_ const pivot = threadIdx.x * 1024;

	uint_ const i = _lower_bound( adj, deg, pivot );
	bounds[threadIdx.x] = i;
	__syncthreads();

	if( threadIdx.x < blockDim.x - 1 )
		pivots[blockIdx.x * ( blockDim.x - 1 ) + threadIdx.x] =
		    i | ( bounds[threadIdx.x + 1] << 16 );
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

    ulong_ * history = nullptr,
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
				ulong_ const hist = ( history[i] << 1 ) | spiked;
				history[i] = hist;

				if( !( hist >> ( delay - 1 ) ) && iter - ages[i] == 63 )
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
    ulong_ * history = nullptr,
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
						auto updt = [&]( int_ const nsteps, bool const pre, bool const post ) {
							Model::synapse::template update(
							    synapse_iter<typename Model::synapse>( isyn ),
							    nsteps,
							    pre,
							    post,
							    dt,
							    info,
							    bak );
						};

						int_ k = iter - ages[src];
						ulong_ hist = history[dst] << ( 63 - k ) >> ( 63 - k );
						while( hist )
						{
							int_ const hsbsub1 = 62 - __clzll( hist );

							updt( k - hsbsub1, MODE == HNDL_SPKS && hsbsub1 == -1, true );

							hist ^= ulong_( 1 ) << ( hsbsub1 + 1 );
							k = hsbsub1;
						}
						updt( k + 1, MODE == HNDL_SPKS && k >= 0, false );
					}

				if constexpr( MODE == HNDL_SPKS )
					Model::neuron::template receive(
					    src,
					    neuron_iter<typename Model::neuron>( dst ),
					    const_synapse_iter<typename Model::synapse>( isyn ),
					    info,
					    bak );
			}
		}

		if constexpr( MODE != INIT_SYNS && Model::synapse::size > 0 )
		{
			__syncthreads();
			if( threadIdx.x == 0 ) ages[src] = iter + 1;
		}
	}
}

template <typename Neuron>
struct shared_iter
{
	char * const _data;
	int_ const _id;

	__device__ shared_iter( char * p, int_ id )
	    : _data( p )
	    , _id( id )
	{
	}

	template <int_ I>
	__device__ auto & get()
	{
		constexpr int_ offset = Neuron::template offset_in_bytes<I>();
		constexpr int_ sz = Neuron::template ith_size_in_bytes<I>();

		return reinterpret_cast<std::tuple_element_t<I, typename Neuron::tuple_t> &>(
		    _data[offset * 1024 + sz * ( _id & 0x3ff )] );
	}

	__device__ int_ id() const { return _id; }
};

template <typename Model>
static __global__ void _gather(
    ulong_ const seed,
    snn_info const info,
    float const dt,
    int_ const delay,
    span2d<int_ const> adj,
    __restrict__ uint_ const * pivots,

    __restrict__ int_ const * in_spikes,
    __restrict__ uint_ const * in_num_spikes,

    __restrict__ int_ * out_spikes,
    __restrict__ uint_ * out_num_spikes )
{
	__shared__ char state[Model::neuron::size_in_bytes * 1024];

	int_ const I = threadid();
	backend bak( I ^ seed );
	int_ const deg_pivots = ( info.num_neurons + 1023 ) / 1024;

	neuron_iter<typename Model::neuron> nit( I );
	shared_iter<typename Model::neuron> sit( state, I );

	if( I < info.num_neurons )
		for_n<Model::neuron::size>( [&]( auto i ) { get<i>( sit ) = get<i>( nit ); } );
	__syncthreads();

	for( int_ d = 0; d < delay; d++ )
	{
		for( int_ s = warpid_block(); s < in_num_spikes[d]; s += WARP_SZ )
		{
			int_ const src = in_spikes[d * info.num_neurons + s];
			uint_ const bounds = pivots[src * deg_pivots + blockIdx.x];

			for( int_ i = laneid() + ( bounds & 0xffff ), last = bounds >> 16; i < last;
			     i += WARP_SZ )
			{
				int_ const dst = adj( src, i );

				Model::neuron::template receive(
				    src,
				    shared_iter<typename Model::neuron>( state, dst ),
				    const_synapse_iter<typename Model::synapse>( -1u ),
				    info,
				    bak );
			}
		}
		__syncthreads();

		if( I < info.num_neurons && Model::neuron::template update( sit, dt, info, bak ) )
			out_spikes[d * info.num_neurons + atomicInc( &out_num_spikes[d], info.num_neurons )] = I;
		__syncthreads();
	}

	if( I < info.num_neurons )
		for_n<Model::neuron::size>( [&]( auto i ) { get<i>( nit ) = get<i>( sit ); } );
}

template <typename T>
static __global__ void _zero_async( T * t )
{
	t[threadIdx.x] = T( 0 );
}


namespace spice
{
namespace cuda
{
void generate_adj_list( cudaStream_t s, spice::util::layout const & desc, int_ * edges )
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

	spice_assert( desc.size() <= ( 1u << 31 ) - 1 );
	spice_assert( desc.max_degree() < SHRT_MAX );

	call( [&] {
		_generate_adj_ids<<<narrow<int>( desc.size() ), WARP_SZ, 0, s>>>(
		    seed(),
		    narrow<int>( desc.connections().size() ),
		    narrow<int>( desc.size() ),
		    narrow<uint_>( desc.max_degree() ),
		    edges );
	} );
}

void generate_pivots(
    cudaStream_t s, int_ const n, int_ const max_degree, int_ const * edges, uint_ * pivots )
{
	spice_assert( ( n + 1023 ) / 1024 + 1 <= 1024) ;
	
	call( [&] {
		_generate_pivots<<<n, ( n + 1023 ) / 1024 + 1, 0, s>>>( edges, max_degree, pivots );
	} );
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

    ulong_ * history /* = {} */,
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
    ulong_ *,
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
    ulong_ *,
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
    ulong_ *,
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
    ulong_ *,
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
    ulong_ * history /* = nullptr */,
    int_ const iter /* = 0 */,
    float const dt /* = 0 */ )
{
	call( [&] {
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
    ulong_ *,
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
    ulong_ *,
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
    ulong_ *,
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
    ulong_ *,
    int_ const iter,
    float const dt );

template <typename Model>
void gather(
    cudaStream_t s,

    snn_info const info,
    float const dt,
    int const delay,
    span2d<int_ const> adj,
    uint_ const * pivots,

    int_ const * in_spikes,
    uint_ const * in_num_spikes,

    int_ * out_spikes,
    uint_ * out_num_spikes )
{
	_gather<Model><<<( info.num_neurons + 1023 ) / 1024, 1024, 0, s>>>(
	    seed(),
	    info,
	    dt,
	    delay,
	    adj,
	    pivots,

	    in_spikes,
	    in_num_spikes,

	    out_spikes,
	    out_num_spikes );
}
template void gather<::spice::synth>(
    cudaStream_t,

    snn_info const,
    float const,
    int_ const,
    span2d<int_ const>,
    uint_ const *,

    int_ const *,
    uint_ const *,

    int_ *,
    uint_ * );
template void gather<::spice::vogels_abbott>(
    cudaStream_t,

    snn_info const,
    float const,
    int_ const,
    span2d<int_ const>,
    uint_ const *,

    int_ const *,
    uint_ const *,

    int_ *,
    uint_ * );
template void gather<::spice::brunel>(
    cudaStream_t,

    snn_info const,
    float const,
    int_ const,
    span2d<int_ const>,
    uint_ const *,

    int_ const *,
    uint_ const *,

    int_ *,
    uint_ * );

template <typename T>
void zero_async( T * t, cudaStream_t s /* = nullptr */, int_ const n /* = 1 */ )
{
	call( [&] { _zero_async<T><<<1, n, 0, s>>>( t ); } );
}
template void zero_async<int>( int_ *, cudaStream_t, int_ const );
template void zero_async<int64_t>( int64_t *, cudaStream_t, int_ const );
template void zero_async<uint_>( uint_ *, cudaStream_t, int_ const );
template void zero_async<uint64_t>( uint64_t *, cudaStream_t, int_ const );
template void zero_async<float>( float *, cudaStream_t, int_ const );
template void zero_async<double>( double *, cudaStream_t, int_ const );
} // namespace cuda
} // namespace spice