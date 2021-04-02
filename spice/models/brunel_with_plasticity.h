#pragma once

#include <spice/models/model.h>
#include <spice/snn_info.h>


namespace spice
{
struct brunel_with_plasticity : model
{
	enum neuron_attr
	{
		V,
		Twait
	};

	enum syn_attr
	{
		W,
		Zpre,
		Zpost
	};

	struct neuron : ::spice::neuron<float, int_>
	{
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
			int_ const Tref = 20;             // dt
			float const Vthres = 0.02f;       // v

			if( n.id() < static_cast<uint_>( info.num_neurons / 2 ) ) // poisson neuron
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
		HYBRID static void receive( int_, Iter dst, SynIter && syn, snn_info info, Backend & bak )
		{
			using util::get;

			if( get<Twait>( dst ) <= 0 )
				bak.atomic_add( get<V>( dst ), get<W>( syn ) * 20'000 / info.num_neurons );
		}
	};

	struct synapse : ::spice::synapse<float, float, float>
	{
		template <typename Iter, typename Backend>
		HYBRID static void init( Iter syn, int_ src, int_, snn_info info, Backend & )
		{
			using util::get;

			auto const Wex = 0.0001f;
			auto const Win = -0.0005f;

			auto const nexc = static_cast<int>( 0.9f * info.num_neurons );

			if( src < nexc )
				get<W>( syn ) = Wex;
			else
				get<W>( syn ) = Win;

			get<Zpre>( syn ) = 1.0f;
			get<Zpost>( syn ) = 1.0f;
		}

		template <typename Iter, typename Backend>
		HYBRID static void update(
		    Iter && syn,
		    int_ const nsteps,
		    bool const pre,
		    bool const post,
		    float const dt,
		    snn_info const info,
		    Backend & bak )
		{
			using util::get;

			float const TstdpInv = 1.0f / 0.02f;
			float const dtInv = 1.0f / dt;

			get<Zpre>( syn ) *= bak.pow( 1 - dt * TstdpInv, nsteps );
			get<Zpost>( syn ) *= bak.pow( 1 - dt * TstdpInv, nsteps );

			get<Zpre>( syn ) += pre;
			get<Zpost>( syn ) += post;

			get<W>( syn ) = bak.clamp(
			    get<W>( syn ) -
			        pre * 0.0202f * get<W>( syn ) * bak.exp( -get<Zpost>( syn ) * dtInv ) +
			        post * 0.01f * ( 1.0f - get<W>( syn ) ) * bak.exp( -get<Zpre>( syn ) * dtInv ),
			    0.0f,
			    0.0003f );
		}

		HYBRID static bool plastic( int_ const src, int_ const dst, snn_info const info )
		{
			auto const npoisson = info.num_neurons / 2;
			auto const nexc = 9 * info.num_neurons / 10;

			return ( src >= npoisson && src < nexc && dst < nexc );
		}
	};
};
} // namespace spice
