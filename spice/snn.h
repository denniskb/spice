#pragma once

#include <spice/snn_info.h>
#include <spice/util/adj_list.h>
#include <spice/util/neuron_group.h>
#include <spice/util/numeric.h>

#include <vector>


namespace spice
{
// Abstract base class for various backend implementations
// such as cpu::snn, cuda::snn, etc.
template <typename Model>
class snn
{
public:
	virtual ~snn() = default;

	void step( std::vector<int> * out_spikes = nullptr );

	std::size_t num_neurons() const;
	std::size_t num_synapses() const;
	float dt() const;
	int delay() const;
	snn_info info() const;

	virtual util::adj_list const & graph() const = 0;
	virtual typename Model::neuron::tuple_t get_neuron( std::size_t i ) const = 0;
	virtual typename Model::synapse::tuple_t get_synapse( std::size_t i ) const = 0;

protected:
	explicit snn( float dt, int delay = 1 );
	void init( float dt, int delay = 1 );

private:
	float _dt = 0.0f;
	int _delay = 1;
	int _i = 0;
	util::kahan_sum<float> _simtime;

	virtual void _step( int i, float dt, std::vector<int> * out_spikes ) = 0;
};
} // namespace spice
