#pragma once

#include <spice/models/model.h>
#include <spice/util/meta.h>


// A synthetic model with a single neuron population, trivial neuron state, fixed connectivity and
// spiking probability. The neuron state (a simple running avg) is required to simulate the effects
// of (cache coherent) neuron updates. For testing purposes.
namespace spice
{
struct synth : model
{
	struct neuron : ::spice::neuron<int>
	{
		enum attr
		{
			N
		};

		template <typename Iter, typename Backend>
		HYBRID static void init( Iter n, snn_info, Backend & )
		{
			using util::get;

			get<N>( n ) = 0;
		}

		template <typename Iter, typename Backend>
		HYBRID static bool update( Iter, float const, snn_info, Backend & bak )
		{
			return bak.rand() < 0.001f;
		}

		template <typename Iter, typename SynIter, typename Backend>
		HYBRID static void receive( int_, Iter dst, SynIter, snn_info, Backend & bak )
		{
			using util::get;

			bak.atomic_add( get<N>( dst ), 1 );
		}
	};
};
} // namespace spice