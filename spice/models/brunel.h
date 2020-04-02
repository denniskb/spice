#pragma once

#include <spice/models/model.h>
#include <spice/snn_info.h>


namespace spice
{
struct brunel : model
{
	struct neuron : ::spice::neuron<float, int>
	{             //                  |     |
		enum attr //                  |     |
		{         //                  |     |
			V,    //__________________|     |
			Twait //________________________|
		};

		template <typename Iter, typename Backend>
		HYBRID static void init( Iter n, snn_info, Backend & )
		{
			using util::get;

			float const Vrest = 0; // v

			get<V>( n ) = Vrest;
			get<Twait>( n ) = 0;
		}

		template <typename Iter, typename Backend>
		HYBRID static bool update( Iter n, float const dt, snn_info info, Backend & bak )
		{
			using util::get;

			float const TmemInv = 1.0 / 0.02; // s
			float const Vrest = 0.0;          // v
			int const Tref = 20;              // dt
			float const Vthres = 0.02f;       // v

			if( n.id() < info.num_neurons / 2 ) // poisson neuron
			{
				float const firing_rate = 20; // Hz

				return bak.rand() < ( firing_rate * dt );
			}
			else
			{
				if( --get<Twait>( n ) <= 0 )
				{
					if( get<V>( n ) > Vthres )
					{
						get<V>( n ) = Vrest;
						get<Twait>( n ) = Tref;
						return true;
					}

					get<V>( n ) += ( Vrest - get<V>( n ) ) * ( dt * TmemInv );
				}
			}

			return false;
		}

		template <typename Iter, typename SynIter, typename Backend>
		HYBRID static void receive( int src, Iter dst, SynIter, snn_info info, Backend & bak )
		{
			using util::get;

			if( get<Twait>( dst ) <= 0 )
			{
				auto const nexc = static_cast<int>( 0.9f * info.num_neurons );
				float const Wex = 0.0001f * 20'000 / info.num_neurons;  // v
				float const Win = -0.0005f * 20'000 / info.num_neurons; // v

				if( src < nexc ) // excitatory src neuron
					bak.atomic_add( get<V>( dst ), Wex );
				else // inhibotory src neuron
					bak.atomic_add( get<V>( dst ), Win );
			}
		}
	};
};
} // namespace spice
