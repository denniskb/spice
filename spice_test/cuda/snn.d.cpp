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
#pragma warning( disable : 4127 ) // "conditional expression is constant"
template <typename Model>
static bool neurons_close( snn<Model> const & lhs, snn<Model> const & rhs, double thres )
{
	if( Model::neuron::size == 0 )
		return true;

	if( lhs.num_neurons() != rhs.num_neurons() )
		return false;

	for( std::size_t i = 0; i < lhs.num_neurons(); i++ )
		if( util::transform_reduce(
		        lhs.get_neuron( i ),
		        rhs.get_neuron( i ),
		        0.0f,
		        []( auto x, auto y ) { return std::abs( x - y ); },
		        []( auto x, auto y ) -> float { return x + y; } ) /
		        Model::neuron::size >
		    thres )
			return false;

	return true;
}

template <typename Model>
static bool synapses_close( snn<Model> const & lhs, snn<Model> const & rhs, double thres )
{
	if( Model::synapse::size == 0 )
		return true;

	if( lhs.num_synapses() != rhs.num_synapses() )
		return false;


	for( std::size_t i = 0; i < lhs.num_synapses(); i++ )
		if( util::transform_reduce(
		        lhs.get_synapse( i ),
		        rhs.get_synapse( i ),
		        0.0f,
		        []( float x, float y ) { return std::abs( x - y ); },
		        []( float x, float y ) -> float { return x + y; } ) /
		        Model::neuron::size >
		    thres )
			return false;

	return true;
}
#pragma warning( pop )

template <typename Model>
static bool graphs_equal( snn<Model> const & lhs, snn<Model> const & rhs )
{
	for( std::size_t i = 0; i < lhs.num_neurons(); i++ )
		if( lhs.graph().neighbors( i ) != rhs.graph().neighbors( i ) )
			return false;

	return true;
}

template <typename Model>
static bool close( snn<Model> const & lhs, snn<Model> const & rhs, double thres )
{
	return graphs_equal( lhs, rhs ) && neurons_close( lhs, rhs, thres ) &&
	       synapses_close( lhs, rhs, thres );
}


TEST_ALL_MODELS( dSNN );

TYPED_TEST( dSNN, Ctor )
{
	{ // Ctor
		cuda::snn<TypeParam> d( 100, 0.1f, 0.0001f );
		ASSERT_EQ( d.num_neurons(), 100 );
		ASSERT_EQ( d.dt(), 0.0001f );
		ASSERT_EQ( d.delay(), 1 );
	}

	{ // Ctor
		cuda::snn<TypeParam> d( 100, 0.1f, 0.0001f, 15 );
		ASSERT_EQ( d.num_neurons(), 100 );
		ASSERT_EQ( d.dt(), 0.0001f );
		ASSERT_EQ( d.delay(), 15 );

		for( std::size_t i = 0; i < 100; i++ )
		{
			int prev = -1;
			for( auto j : d.graph().neighbors( i ) )
			{
				ASSERT_GE( j, 0 );
				ASSERT_LT( j, 100 );

				ASSERT_GT( j, prev );
				prev = j;
			}
		}

		cpu::snn<TypeParam> h( 100, 0.1f, 0.0001f );
		ASSERT_TRUE( neurons_close( h, d, 0.0 ) );
	}

	{ // Conv. Ctor
		cpu::snn<TypeParam> h( 100, 0.1f, 0.0001f, 15 );
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
	cpu::snn<vogels_abbott> h( 4000, 0.02f, 0.0001f );
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
	cpu::snn<vogels_abbott> h( 4000, 0.02f, 0.0001f, 8 );
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