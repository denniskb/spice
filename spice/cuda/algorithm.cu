#include "algorithm.h"

#include <spice/cuda/backend.cuh>
#include <spice/cuda/util/error.h>
#include <spice/cuda/util/random.cuh>
#include <spice/cuda/util/utility.cuh>
#include <spice/cuda/util/warp.cuh>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/assert.h>
#include <spice/util/circular_buffer.h>

#include <array>


using namespace spice;
using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


__constant__ int2 _desc[20];
__constant__ void * _neuron_storage[20];
__constant__ void * _synapse_storage[20];
static unsigned _seed = 0;


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
    int const desc_len, int const N, int const * layout, int * const out_edges, int const width )
{
	int const MAX_DEGREE = 1024;
	__shared__ float rows[4][MAX_DEGREE];

	xorwow rng( threadid() );

	for( unsigned wid = warpid_grid(); wid < N; wid += num_warps() )
	{
		unsigned offset = wid * width;
		for( int c = 0; c < desc_len; c++ )
		{
			int degree = layout[wid + c * N];
			int first = _desc[c].x;
			int range = _desc[c].y;

			while( degree > 0 )
			{
				int const d = min( degree, MAX_DEGREE );
				int const r = (int)( (long long)d * range / degree );

				// accumulate
				float total = 0.0f;
				for( int i = laneid(); i < d; i += WARP_SZ )
				{
					float f = -logf( rng() );

					float sum;
					f = total + warp::inclusive_scan( f, sum, __activemask() );
					total += sum;

					rows[warpid_block()][i] = f;
				}

				// normalize
				{
					total -= logf( rng() );
					total = __shfl_sync( MASK_ALL, total, 0 );

					float const scale = ( r - d ) / total;
					for( int i = laneid(); i < d; i += WARP_SZ )
						( out_edges + offset )[i] =
						    first + static_cast<int>( rows[warpid_block()][i] * scale ) + i;
				}

				offset += d;
				degree -= MAX_DEGREE;
				first += r;
				range -= r;
			}
		}

		for( unsigned i = offset + laneid(); i < ( wid + 1 ) * width; i += WARP_SZ )
			out_edges[i] = -1;
	}
}

template <typename Model, bool INIT>
static __global__ void _process_neurons(
    snn_info const info,
    unsigned const seed,

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
	assert( info.num_neurons < INT_MAX - num_threads() );

	backend bak( threadid() + num_threads() * seed );

	for( int i = threadid(); i < info.num_neurons; i += num_threads() )
	{
		neuron_iter<typename Model::neuron> it( i );

		if( INIT )
			Model::neuron::template init( it, info, bak );
		else // udpate
		{
			bool const spiked = Model::neuron::template update( it, dt, info, bak );

			if( Model::synapse::size > 0 ) // plast.
			{
				unsigned const flag = __ballot_sync( __activemask(), spiked );
				if( laneid() == 0 ) history[i / WARP_SZ] = flag;

				bool const delayed_spike = delayed_history[i / WARP_SZ] >> ( i % WARP_SZ ) & 1u;

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
    unsigned const seed,
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
	backend bak( threadid() + seed * num_threads() );

	for( int i = blockIdx.x; i < ( ( MODE == INIT_SYNS ) ? info.num_neurons : *num_spikes );
	     i += gridDim.x )
	{
		unsigned const src = ( MODE == INIT_SYNS ) ? i : spikes[i];

		for( unsigned j = threadIdx.x; j < adj.width(); j += blockDim.x )
		{
			unsigned const isyn = adj.row( src ) - adj.row( 0 ) + j; // src * max_degree + j;
			int const dst = adj( src, j );

			if( dst >= 0 )
			{
				if( MODE == INIT_SYNS )
					Model::synapse::template init(
					    synapse_iter<typename Model::synapse>( isyn ), src, dst, info, bak );
				else if( Model::synapse::size > 0 )
					for( int k = ages[src]; k <= iter; k++ )
						Model::synapse::template update(
						    synapse_iter<typename Model::synapse>( isyn ),
						    src,
						    dst,
						    MODE == HNDL_SPKS && k == iter,
						    history( circidx( k, max_history ), dst / WARP_SZ ) >>
						            ( dst % WARP_SZ ) &
						        1u,
						    dt,
						    info,
						    bak );

				if( MODE == HNDL_SPKS )
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
    unsigned const seed,
    span2d<int const> adj,

    int const * spikes = nullptr,
    unsigned const * num_spikes = nullptr )
{
	backend bak( threadid() + seed * num_threads() );

	int const S = *num_spikes;

	for( int i = warpid_grid(); i < S * ( adj.width() / WARP_SZ ); i += num_warps() )
	{
		int s = i % S;
		int o = i / S;

		int src = spikes[s];
		int dst = adj( src, WARP_SZ * o + laneid() );

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


template <typename Model>
static void
_upload_meta( Model::neuron::ptuple_t const & neuron, Model::synapse::ptuple_t const & synapse )
{
	static_assert(
	    Model::neuron::size <= 20,
	    "spice doesn't support models with more than 20 neuron attributes" );
	static_assert(
	    Model::synapse::size <= 20,
	    "spice doesn't support models with more than 20 synapse attributes" );

	if( Model::neuron::size > 0 )
	{
		std::array<void *, Model::neuron::size> tmp;
		spice::util::for_each_i( neuron, [&]( auto p, int i ) { tmp[i] = p; } );

		success_or_throw(
		    cudaMemcpyToSymbolAsync( _neuron_storage, tmp.data(), sizeof( void * ) * tmp.size() ) );
	}

	if( Model::synapse::size > 0 )
	{
		std::array<void *, Model::synapse::size> tmp;
		spice::util::for_each_i( synapse, [&]( auto p, int i ) { tmp[i] = p; } );

		success_or_throw( cudaMemcpyToSymbolAsync(
		    _synapse_storage, tmp.data(), sizeof( void * ) * tmp.size() ) );
	}
}


namespace spice
{
namespace cuda
{
void generate_rnd_adj_list(
    spice::util::neuron_group const & desc, int const * layout, int * out_edges, int const width )
{
	spice_assert(
	    desc.connections().size() <= 20,
	    "spice doesn't support models with more than 20 connections between neuron populations" );

	std::array<int2, 20> tmp;
	for( std::size_t i = 0; i < desc.connections().size(); i++ )
	{
		tmp[i].x = desc.first( std::get<1>( desc.connections().at( i ) ) );
		tmp[i].y = desc.range( std::get<1>( desc.connections().at( i ) ) );
	}
	success_or_throw(
	    cudaMemcpyToSymbolAsync( _desc, tmp.data(), sizeof( int2 ) * desc.connections().size() ) );

	_generate_adj_ids<<<128, 128>>>(
	    desc.connections().size(), desc.size(), layout, out_edges, width );
}

template <typename Model>
void upload_meta( Model::neuron::ptuple_t const & neuron, Model::synapse::ptuple_t const & synapse )
{
	_upload_meta<Model>( neuron, synapse );
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
	_process_neurons<Model, true><<<nblocks( info.num_neurons, 128, 256 ), 256>>>( info, _seed++ );

	if( Model::synapse::size > 0 )
		_process_spikes<Model, INIT_SYNS><<<128, 256>>>( info, _seed++, adj );
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
	_process_neurons<Model, false><<<nblocks( info.num_neurons, 128, 256 ), 256>>>(
	    info,
	    _seed++,
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

	if( Model::synapse::size > 0 )
		_process_spikes<Model, UPDT_SYNS><<<256, 256>>>(
		    info,
		    _seed++,
		    adj,

		    updates,
		    num_updates,

		    ages,
		    history,
		    max_history,
		    iter,
		    delay,
		    dt );
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
	int const launch = Model::synapse::size > 0 ? 256 : 128;

	// TODO: Fine-tune boundary for final paper!
	if( Model::synapse::size > 0 || info.num_neurons < 400'000 )
		_process_spikes<Model, HNDL_SPKS><<<launch, launch>>>(
		    info,
		    _seed++,
		    adj,

		    spikes,
		    num_spikes,

		    ages,
		    history,
		    max_history,
		    iter,
		    delay,
		    dt );
	else
		_process_spikes_cache_aware<Model><<<launch, launch>>>(
		    info,
		    _seed++,
		    adj,

		    spikes,
		    num_spikes );
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
	_zero_async<T><<<1, 1, 0, s>>>( t );
}
template void zero_async<int>( int *, cudaStream_t );
template void zero_async<int64_t>( int64_t *, cudaStream_t );
template void zero_async<unsigned>( unsigned *, cudaStream_t );
template void zero_async<uint64_t>( uint64_t *, cudaStream_t );
template void zero_async<float>( float *, cudaStream_t );
template void zero_async<double>( double *, cudaStream_t );
} // namespace cuda
} // namespace spice