#include <spice/cuda/multi_snn.h>

#include <spice/cuda/util/device.h>
#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>


using namespace spice::util;
using namespace spice::cuda::util;


namespace spice::cuda
{
template <typename Model>
multi_snn<Model>::multi_snn( spice::util::layout desc, float dt, int delay /* = 1 */ )
    : _dt( dt )
    , _delay( delay )
{
	for( auto & d : device::devices() )
	{
		d.set();
		_nets.push_back( snn<Model>(
		    slice( desc, device::devices().size(), d ),
		    dt,
		    delay,
		    narrow_int<int>( device::devices().size() ),
		    d ) );
	}
}

template <typename Model>
void multi_snn<Model>::step()
{
	for( auto & d : device::devices() )
	{
		d.set();
		// TODO: Tradeoff: fewer cudaSetDevice()s vs blocking spike sync
		// for( int i = 0; i < _nets[d].delay(); i++ )
		_nets[d].step();
		// TODO: Exchange spikes
	}
}

template <typename Model>
void multi_snn<Model>::sync()
{
	for( auto & d : device::devices() )
	{
		d.set();
		success_or_throw( cudaDeviceSynchronize() );
	}
}

// static
template <typename Model>
layout multi_snn<Model>::slice( layout const & whole, std::size_t n, std::size_t i )
{
	int const first = narrow_int<int>( whole.size() * i / n );
	int const last = narrow_int<int>( whole.size() * ( i + 1 ) / n );

	std::vector<layout::edge> part;
	for( auto c : whole.connections() )
	{
		std::get<2>( c ) = std::max( first, std::get<2>( c ) );
		std::get<3>( c ) = std::min( last, std::get<3>( c ) );
		if( std::get<2>( c ) < std::get<3>( c ) ) part.push_back( std::move( c ) );
	}

	return layout( whole.size(), part );
}

template class multi_snn<spice::vogels_abbott>;
template class multi_snn<spice::brunel>;
template class multi_snn<spice::brunel_with_plasticity>;
template class multi_snn<spice::synth>;
} // namespace spice::cuda