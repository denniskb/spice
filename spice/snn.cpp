#include "snn.h"

#include <spice/models/brunel.h>
#include <spice/models/brunel_with_plasticity.h>
#include <spice/models/synth.h>
#include <spice/models/vogels_abbott.h>
#include <spice/util/assert.h>
#include <spice/util/type_traits.h>


namespace spice
{
template <typename Model>
snn<Model>::snn( float dt, int delay /* = 1 */ )
{
	spice_assert( dt >= 0.0f );
	spice_assert( delay >= 1 );

	init( dt, delay );
}


template <typename Model>
void snn<Model>::step( std::vector<int> * out_spikes /* = nullptr */ )
{
	_step( _i++, _simtime.add( dt() ), out_spikes );
}


template <typename Model>
std::size_t snn<Model>::num_neurons() const
{
	return graph().num_nodes();
}
template <typename Model>
std::size_t snn<Model>::num_synapses() const
{
	return graph().num_edges();
}
template <typename Model>
float snn<Model>::dt() const
{
	return _dt;
}
template <typename Model>
int snn<Model>::delay() const
{
	return _delay;
}
template <typename Model>
snn_info snn<Model>::info() const
{
	return {util::narrow_cast<int>( num_neurons() )};
}

template <typename Model>
void snn<Model>::init( float dt, int delay )
{
	spice_assert( dt >= 0.0f );
	spice_assert( delay >= 1 );

	_dt = dt;
	_delay = delay;
}


template class snn<vogels_abbott>;
template class snn<brunel>;
template class snn<brunel_with_plasticity>;
template class snn<synth>;
} // namespace spice
