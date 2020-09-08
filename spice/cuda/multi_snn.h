#pragma once

#include <spice/cuda/snn.h>
#include <spice/cuda/util/device.h>
#include <spice/cuda/util/event.h>
#include <spice/cuda/util/stream.h>
#include <spice/snn.h>
#include <spice/util/span.hpp>

#include <array>
#include <atomic>
#include <mutex>
#include <thread>


namespace spice::cuda
{
template <typename Model>
class multi_snn : public ::spice::snn<Model>
{
public:
	multi_snn( spice::util::layout desc, float dt, int_ delay = 1 );
	multi_snn( spice::snn<Model> const & net );
	~multi_snn();

	void step( std::vector<int> * out_spikes = nullptr ) override;
	void sync();

	size_ num_neurons() const override;
	size_ num_synapses() const override;

	std::pair<std::vector<int>, size_> adj() const override;
	std::vector<typename Model::neuron::tuple_t> neurons() const override;
	std::vector<typename Model::synapse::tuple_t> synapses() const override;

private:
	std::array<std::optional<cuda::snn<Model>>, util::device::max_devices> _nets;
	struct
	{
		std::unique_ptr<uint_, cudaError_t ( * )( void * )> counts_data;
		nonstd::span<uint_> counts;

		std::array<int_ *, util::device::max_devices> ddata;
		std::array<uint_ *, util::device::max_devices> dcounts;
	} _spikes;

	std::array<std::optional<util::stream>, util::device::max_devices> _cp;
	std::array<std::optional<util::sync_event>, util::device::max_devices> _updt;

	std::array<std::thread, util::device::max_devices> _workers;
	std::atomic_bool _running{ true };
	std::atomic_int32_t _iter{ 0 };
	std::atomic_int32_t _work{ 0 };
	std::vector<int_> * _out_spikes = nullptr;
	std::mutex _out_spikes_lock;

	multi_snn( float dt, int_ delay );
};
} // namespace spice::cuda