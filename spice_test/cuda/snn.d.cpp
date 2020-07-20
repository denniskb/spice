#include <gtest/gtest.h>

#include "../model.h"

#include <spice/cpu/snn.h>
#include <spice/cuda/snn.h>
#include <spice/models/vogels_abbott.h>

#include <algorithm>


using namespace spice;


template <typename Cont1, typename Cont2>
bool set_equal( Cont1 const & lhs, Cont2 const & rhs )
{
	return std::is_same_v<Cont1::value_type, Cont2::value_type> &&
	       std::distance( lhs.begin(), lhs.end() ) == std::distance( rhs.begin(), rhs.end() ) &&
	       std::is_permutation( lhs.begin(), lhs.end(), rhs.begin() );
}

#pragma warning( push )
#pragma warning( disable : 4127 4723 ) // "conditional expression is constant"
template <typename Pop>
static bool close( Pop const & lhs, Pop const & rhs, double thres )
{
	constexpr auto sz = std::tuple_size_v<Pop::value_type>;
	if( sz == 0 ) return true;

	if( lhs.size() != rhs.size() ) return false;

	for( std::size_t i = 0; i < lhs.size(); i++ )
		if( util::reduce(
		        util::map( lhs[i], rhs[i], []( auto x, auto y ) { return std::abs( x - y ); } ),
		        0,
		        []( auto x, auto y ) { return x + y; } ) /
		        sz >
		    thres )
			return false;

	return true;
}
#pragma warning( pop )

template <typename Model>
static bool graphs_equal( snn<Model> const & lhs, snn<Model> const & rhs )
{
	if( lhs.num_neurons() != rhs.num_neurons() || lhs.num_synapses() != rhs.num_synapses() )
		return false;

	// TOOD: Rewrite to be more generic by wrapping lhs/rhs into adj_list
	auto _lhs = lhs.graph();
	auto _rhs = rhs.graph();

	if( _lhs.second != _rhs.second ) return false;

	return _lhs.first == _rhs.first;
}

template <typename Model>
static bool close( snn<Model> const & lhs, snn<Model> const & rhs, double thres )
{
	return graphs_equal( lhs, rhs ) && close( lhs.neurons(), rhs.neurons(), thres ) &&
	       close( lhs.synapses(), rhs.synapses(), thres );
}


TEST_ALL_MODELS( dSNN );

TYPED_TEST( dSNN, Ctor )
{
	{ // Ctor
		cuda::snn<TypeParam> d( { 100, 0.1f }, 0.0001f );
		ASSERT_EQ( d.num_neurons(), 100 );
		ASSERT_EQ( d.dt(), 0.0001f );
		ASSERT_EQ( d.delay(), 1 );
	}

	{ // Ctor
		cuda::snn<TypeParam> d( { 100, 0.1f }, 0.0001f, 15 );
		ASSERT_EQ( d.num_neurons(), 100 );
		ASSERT_EQ( d.dt(), 0.0001f );
		ASSERT_EQ( d.delay(), 15 );

		cpu::snn<TypeParam> h( { 100, 0.1f }, 0.0001f );
		ASSERT_TRUE( close( h.neurons(), d.neurons(), 0.0 ) );
	}

	{ // Conv. Ctor
		cpu::snn<TypeParam> h( { 100, 0.1f }, 0.0001f, 15 );
		cuda::snn d( h );
		ASSERT_EQ( d.num_neurons(), 100 );
		ASSERT_EQ( d.dt(), 0.0001f );
		ASSERT_EQ( d.delay(), 15 );

		ASSERT_TRUE( close( h, d, 0.0 ) );
	}
}

// Can't test brunel due to randomization in update step (poisson firing rate). TODO?: Find way to
// draw same numbers on CPU and GPU/deactivate rng for testing purposes..
TEST( dSNN, Step )
{
	cpu::snn<vogels_abbott> h( { 4000, 0.02f }, 0.0001f );
	cuda::snn d( h );
	ASSERT_TRUE( close( h, d, 0.0 ) );

	std::vector<int> h_spikes, d_spikes;

	for( int i = 0; i < 1000; i++ )
	{
		h.step( &h_spikes );
		d.step( &d_spikes );
		ASSERT_TRUE( set_equal( h_spikes, d_spikes ) ) << i;
	}

	ASSERT_TRUE( close( h, d, 1e-5 ) );
}

TEST( dSNN, StepWithDelay )
{
	cpu::snn<vogels_abbott> h( { 4000, 0.02f }, 0.0001f, 8 );
	cuda::snn d( h );
	ASSERT_TRUE( close( h, d, 0.0 ) );

	std::vector<int> h_spikes, d_spikes;

	for( int i = 0; i < 1000; i++ )
	{
		h.step( &h_spikes );
		d.step( &d_spikes );
		ASSERT_TRUE( set_equal( h_spikes, d_spikes ) ) << i;
	}

	ASSERT_TRUE( close( h, d, 1e-5 ) );
}