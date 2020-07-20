#pragma once

#include <spice/snn_info.h>
#include <spice/util/adj_list.h>
#include <spice/util/layout.h>
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

	virtual std::size_t num_neurons() const = 0;
	virtual std::size_t num_synapses() const = 0;
	float dt() const;
	int delay() const;
	snn_info info() const;

	// TODO: return variants instead (cpu::snn returns span, gpu::snn returns vector)
	// (edges, width)
	virtual std::pair<std::vector<int>, std::size_t> graph() const = 0;
	virtual std::vector<typename Model::neuron::tuple_t> neurons() const = 0;
	virtual std::vector<typename Model::synapse::tuple_t> synapses() const = 0;

protected:
	explicit snn( float dt, int delay = 1 );

private:
	float const _dt;
	int const _delay;
	int _i = 0;
	util::kahan_sum<float> _simtime;

	virtual void _step( int i, float dt, std::vector<int> * out_spikes ) = 0;
};
} // namespace spice
