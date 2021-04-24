#pragma once

#include <spice/snn_info.h>
#include <spice/util/host_defines.h>
#include <spice/util/meta.h>


namespace spice
{
template <typename... Ts>
struct neuron : util::type_list<Ts...>
{
	// optional if layout empty (i.e. 'myneuron : neuron<>')
	template <typename Iter, typename Backend>
	HYBRID static void init( Iter, snn_info, Backend & )
	{
	}

	template <typename Iter, typename Backend>
	HYBRID static bool update( Iter, float, snn_info, Backend & )
	{
		return false;
	}

	template <typename Iter, typename SynIter, typename Backend>
	HYBRID static void receive( int_, Iter, SynIter &&, snn_info, Backend & )
	{
	}
};

template <typename... Ts>
struct synapse : util::type_list<Ts...>
{
	template <typename Iter, typename Backend>
	HYBRID static void init( Iter, int_, int_, snn_info, Backend & )
	{
	}

	template <typename Iter, typename Backend>
	HYBRID static void
	update( Iter &&, int_ const, bool const, bool const, float const, snn_info const, Backend & )
	{
	}

	HYBRID static bool plastic( int_ const, int_ const, snn_info const ) { return true; }
};

struct model
{
	struct neuron : ::spice::neuron<>
	{
	};

	// optional
	struct synapse : ::spice::synapse<>
	{
	};
};
} // namespace spice