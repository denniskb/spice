#pragma once

#include <spice/models/model.h>
#include <spice/snn_info.h>


namespace spice
{
struct vogels_abbott : model
{
	struct neuron : ::spice::neuron<float, float, float, int>
	{             //                  |      |      |     |
		enum attr //                  |      |      |     |
		{         //                  |      |      |     |
			V,    //__________________|      |      |     |
			Gex,  //_________________________|      |     |
			Gin,  //________________________________|     |
			Twait //______________________________________|
		};

		// TODO: replace FAT backend with on-demand (lazy-eval) one
		template <typename Iter, typename Backend>
		HYBRID static void init( Iter n, snn_info, Backend & )
		{
			using util::get;

			float const Vrest = -0.06f; // v

			get<V>( n ) = Vrest;
			get<Gex>( n ) = 0.0f;
			get<Gin>( n ) = 0.0f;
			get<Twait>( n ) = 0;
		}

		template <typename Iter, typename Backend>
		HYBRID static bool update( Iter n, float const dt, snn_info, Backend & )
		{
			using util::get;

			int const Tref = 50;                // dt
			float const Vrest = -0.06f;         // v
			float const Vthres = -0.05f;        // v
			float const TmemInv = 1.0f / 0.02f; // s
			float const Eex = 0.0f;             // v
			float const Ein = -0.08f;           // v
			float const Ibg = 0.02f;            // v

			float const TexInv = 1.0f / 0.005f; // s
			float const TinInv = 1.0f / 0.01f;  // s

			bool spiked = false;
			if( --get<Twait>( n ) <= 0 )
			{
				if( get<V>( n ) > Vthres )
				{
					get<V>( n ) = Vrest;
					get<Twait>( n ) = Tref;
					spiked = true;
				}
				else
					get<V>( n ) +=
					    ( ( Vrest - get<V>( n ) ) + get<Gex>( n ) * ( Eex - get<V>( n ) ) +
					      get<Gin>( n ) * ( Ein - get<V>( n ) ) + Ibg ) *
					    ( dt * TmemInv );
			}

			get<Gex>( n ) -= get<Gex>( n ) * ( dt * TexInv );
			get<Gin>( n ) -= get<Gin>( n ) * ( dt * TinInv );

			return spiked;
		}

		template <typename Iter, typename SynIter, typename Backend>
		HYBRID static void receive( int src, Iter dst, SynIter, snn_info info, Backend & bak )
		{
			using util::get;

			float const Wex =
			    0.4f * 16'000'000 / ( (long long)info.num_neurons * info.num_neurons ); // siemens
			float const Win =
			    5.1f * 16'000'000 / ( (long long)info.num_neurons * info.num_neurons ); // siemens

			auto const nexc = static_cast<int>( 0.8f * info.num_neurons );

			if( src < nexc ) // excitatory src neuron
				bak.atomic_add( get<Gex>( dst ), Wex );
			else // inhibotory src neuron
				bak.atomic_add( get<Gin>( dst ), Win );
		}
	};
};
} // namespace spice