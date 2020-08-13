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

		auto slice = desc.cut( device::devices().size(), d );
		_nets.push_back( snn<Model>( slice.part, dt, delay, slice.first, slice.last ) );
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

template class multi_snn<spice::vogels_abbott>;
template class multi_snn<spice::brunel>;
template class multi_snn<spice::brunel_with_plasticity>;
template class multi_snn<spice::synth>;
} // namespace spice::cuda