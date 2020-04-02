#include "algorithm.h"

#include <spice/cuda/backend.cuh>
#include <spice/cuda/util/error.h>
#include <spice/cuda/util/random.cuh>
#include <spice/cuda/util/utility.cuh>
#include <spice/cuda/util/warp.cuh>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/circular_buffer.h>

#include <array>


using namespace spice;
using namespace spice::util;
using namespace spice::cuda;
using namespace spice::cuda::util;


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

	__device__ int id() const { return _i; }

private:
	int const _i = 0;
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


template <typename T>
static __global__ void _fill( T * x, int len, T val )
{
	for( int i = threadid(); i < len; i += num_threads() )
		x[i] = val;
}

static __global__ void
_generate_adj_ids( int4 const * const layout, int const len, int * const out_edges )
{
	int const MAX_DEGREE = 1024;
	__shared__ float rows[4][MAX_DEGREE];

	xorwow rng( threadid() );

	for( int wid = warpid_grid(); wid < len; wid += num_warps() )
	{
		int4 desc = layout[wid];

		// TODO: Split into two kernels???
		if( desc.z == -1 )
			for( int i = laneid(); i < desc.y; i += WARP_SZ )
				( out_edges + desc.x )[i] = INT_MAX;
		else
			while( desc.y > 0 )
			{
				int const degree = min( desc.y, MAX_DEGREE );
				int2 const bounds = {desc.z, (float)degree / desc.y * desc.w};

				// accumulate
				float total = 0.0f;
				for( int i = laneid(); i < degree; i += WARP_SZ )
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

					float const scale = ( bounds.y - degree ) / total;
					for( int i = laneid(); i < degree; i += WARP_SZ )
						( out_edges + desc.x )[i] =
						    bounds.x + static_cast<int>( rows[warpid_block()][i] * scale ) + i;
				}

				desc.x += degree;
				desc.y -= MAX_DEGREE;
				desc.z = bounds.x + bounds.y;
				desc.w -= bounds.y;
			}
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
    int const max_history = 0 )
{
	backend bak( threadid() + num_threads() * seed );

	for( int i = threadid(); i < info.num_neurons; i += num_threads() )
	{
		neuron_iter<typename Model::neuron> it( i );

		if( INIT )
			Model::neuron::template init( it, info, bak );
		else // udpate
		{
			bool const spiked = Model::neuron::template update( it, dt, info, bak );

			if( Model::synapse::size > 0 )
			{
				unsigned const flag = __ballot_sync( __activemask(), spiked );
				if( laneid() == 0 )
					history[i / 32] = flag;

				if( iter - ages[i] + delay == max_history - 1 && !spiked )
					updates[atomicInc( num_updates, info.num_neurons )] = i;
			}

			if( spiked )
				spikes[atomicInc( num_spikes, info.num_neurons )] = i;
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
		int const src = ( MODE == INIT_SYNS ) ? i : spikes[i];

		for( int j = threadIdx.x; j < adj.width(); j += blockDim.x )
		{
			int const isyn = adj.row( src ) - adj.row( 0 ) + j; // src * max_degree + j;
			int const dst = adj( src, j );

			if( dst != INT_MAX )
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
						    // history( circidx( k - delay, max_history ), src / 32 ) >> ( src % 32
						    // ) &
						    //    1u,
						    k == iter,
						    history( circidx( k, max_history ), dst / 32 ) >> ( dst % 32 ) & 1u,
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

			if( threadIdx.x == 0 )
				ages[src] = iter + 1;
		}
	}
}

template <typename T>
static __global__ void _zero_async( T * t )
{
	*t = T( 0 );
}


template <typename Model>
static void
_upload_storage( Model::neuron::ptuple_t const & neuron, Model::synapse::ptuple_t const & synapse )
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
    spice::util::adj_list::int4 const * layout,
    int len,
    int num_neurons,
    int max_degree,
    int * out_edges )
{
	_generate_adj_ids<<<128, 128>>>( reinterpret_cast<int4 const *>( layout ), len, out_edges );
}

template <typename Model>
void upload_storage(
    Model::neuron::ptuple_t const & neuron, Model::synapse::ptuple_t const & synapse )
{
	_upload_storage<Model>( neuron, synapse );
}
template void upload_storage<::spice::vogels_abbott>(
    ::spice::vogels_abbott::neuron::ptuple_t const & neuron,
    ::spice::vogels_abbott::synapse::ptuple_t const & synapse );
template void upload_storage<::spice::brunel>(
    ::spice::brunel::neuron::ptuple_t const & neuron,
    ::spice::brunel::synapse::ptuple_t const & synapse );
template void upload_storage<::spice::brunel_with_plasticity>(
    ::spice::brunel_with_plasticity::neuron::ptuple_t const & neuron,
    ::spice::brunel_with_plasticity::synapse::ptuple_t const & synapse );

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
	    max_history );

	if( Model::synapse::size > 0 )
		_process_spikes<Model, UPDT_SYNS><<<nblocks( 5 * info.num_neurons, 128, 256 ), 256>>>(
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
	// Assume 2% neurons spiking
	_process_spikes<Model, HNDL_SPKS><<<nblocks( 5 * info.num_neurons, 128, 256 ), 256>>>(
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