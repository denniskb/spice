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
void multi_snn<Model>::step( std::vector<int> * out_spikes /* = nullptr */ )
{
	int * spikes[8];
	unsigned * n_spikes[8];

	for( auto & d : device::devices() )
	{
		d.set();
		_nets[d].step( &spikes[d], &n_spikes[d] );
	}

	for( auto & d : device::devices() ) d.synchronize();

	// 2 gpus for now:
	int a, b;
	success_or_throw( cudaMemcpy( &a, n_spikes[0], 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( &b, n_spikes[1], 4, cudaMemcpyDefault ) );

	success_or_throw( cudaMemcpy( spikes[0] + a, spikes[1], 4 * b, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( spikes[1] + b, spikes[0], 4 * a, cudaMemcpyDefault ) );

	if( out_spikes )
	{
		out_spikes->resize( a + b );
		cudaMemcpy( out_spikes->data(), spikes[0], 4 * a, cudaMemcpyDefault );
		cudaMemcpy( out_spikes->data() + a, spikes[1], 4 * b, cudaMemcpyDefault );
	}

	a += b;
	success_or_throw( cudaMemcpy( n_spikes[0], &a, 4, cudaMemcpyDefault ) );
	success_or_throw( cudaMemcpy( n_spikes[1], &a, 4, cudaMemcpyDefault ) );
}

template <typename Model>
void multi_snn<Model>::sync()
{
	for( auto & d : device::devices() ) d.synchronize();
}

template <typename Model>
std::size_t multi_snn<Model>::num_neurons() const
{
	return _nets[0].num_neurons();
}

template class multi_snn<spice::vogels_abbott>;
template class multi_snn<spice::brunel>;
template class multi_snn<spice::brunel_with_plasticity>;
template class multi_snn<spice::synth>;
} // namespace spice::cuda