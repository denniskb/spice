#include <benchmark/benchmark.h>

#include <spice_bench/exp_range.h>

#include <spice/cuda/algorithm.h>
#include <spice/cuda/snn.h>
#include <spice/cuda/util/dbuffer.h>
#include <spice/cuda/util/event.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/adj_list.h>
#include <spice/util/type_traits.h>


using namespace spice;
using namespace spice::cuda;
using namespace spice::cuda::util;
using namespace spice::util;


static void gen( benchmark::State & state )
{
	std::size_t const NSYN = state.range( 0 );
	std::size_t const N = narrow_cast<std::size_t>( std::sqrt( NSYN / 0.1 ) );

	state.counters["num_neurons"] = narrow_cast<double>( N );
	state.counters["num_syn"] = narrow_cast<double>( NSYN );

	try
	{
		layout desc( { N }, { { 0, 0, 0.1f } } );

		std::vector<int> l;
		adj_list::generate( desc, l );

		for( auto _ : state ) adj_list::generate( desc, l );
	}
	catch( std::exception & e )
	{
		std::printf( "%s\n", e.what() );
		return;
	}

	state.SetBytesProcessed( ( N * 4 ) * state.iterations() );
}
BENCHMARK( gen )->Unit( benchmark::kMillisecond )->ExpRange( 1'000'000, 2048'000'000 );

static void plot0_AdjGen( benchmark::State & state )
{
	float const P = 0.1f;
	std::size_t const NSYN = state.range( 0 );
	std::size_t const N = narrow_cast<std::size_t>( std::sqrt( NSYN / P ) );

	state.counters["num_neurons"] = narrow_cast<double>( N );
	state.counters["num_syn"] = narrow_cast<double>( NSYN );

	try
	{
		layout desc( { N }, { { 0, 0, P } } );

		dbuffer<int> e( desc.size() * desc.max_degree() );

		event start, stop;
		for( auto _ : state )
		{
			generate_rnd_adj_list( desc, e.data() );
			start.record();
			for( int i = 0; i < 10; i++ ) generate_rnd_adj_list( desc, e.data() );
			// cudaMemsetAsync( e.data(), 0, 4 * NSYN );
			stop.record();
			stop.synchronize();

			state.SetIterationTime( stop.elapsed_time( start ) * 0.001 / 10 );
		}
	}
	catch( std::exception & e )
	{
		std::printf( "%s\n", e.what() );
		return;
	}

	state.SetBytesProcessed( ( NSYN * 4 ) * state.iterations() );
}
BENCHMARK( plot0_AdjGen )
    ->UseManualTime()
    ->Unit( benchmark::kMillisecond )
    ->ExpRange( 1'000'000, 512'000'000 );


// Absolute runtime per iteration as a function of synapse count
template <typename Model>
static void plot2_RunTime( benchmark::State & state )
{
	float const MUL = std::is_same_v<Model, vogels_abbott> ? 0.02f : 0.05f;
	std::size_t const NSYN = state.range( 0 );
	std::size_t const N = narrow_cast<std::size_t>( std::sqrt( NSYN / MUL ) );
	std::size_t const ITER = 1000;

	state.counters["num_neurons"] = narrow_cast<double>( N );
	state.counters["num_syn"] = narrow_cast<double>( NSYN );

	try
	{
		std::unique_ptr<cuda::snn<Model>> net;
		if( std::is_same_v<Model, vogels_abbott> )
			net.reset( new cuda::snn<Model>( { N, 0.02f }, 0.0001f, 8 ) );
		else
			net.reset( new cuda::snn<Model>(
			    layout( { N / 2, N / 2 }, { { 0, 1, 0.1f }, { 1, 1, 0.1f } } ), 0.0001f, 15 ) );

		event start, stop;
		for( auto _ : state )
		{
			for( int i = 0; i < 150; i++ ) net->step();
			start.record();
			for( int i = 0; i < ITER; i++ ) net->step();
			stop.record();
			stop.synchronize();

			state.SetIterationTime( stop.elapsed_time( start ) * 0.001 / ITER );
		}
	}
	catch( std::exception & e )
	{
		std::printf( "%s\n", e.what() );
		return;
	}
}
BENCHMARK_TEMPLATE( plot2_RunTime, vogels_abbott )
    ->UseManualTime()
    ->Unit( benchmark::kMicrosecond )
    ->ExpRange( 125'000, 512'000'000 );
BENCHMARK_TEMPLATE( plot2_RunTime, brunel )
    ->UseManualTime()
    ->Unit( benchmark::kMicrosecond )
    ->ExpRange( 125'000, 512'000'000 );
BENCHMARK_TEMPLATE( plot2_RunTime, brunel_with_plasticity )
    ->UseManualTime()
    ->Unit( benchmark::kMicrosecond )
    ->ExpRange( 125'000, 128'000'000 );