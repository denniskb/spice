#pragma once

#include <spice/cuda/snn.h>
#include <spice/cuda/util/device.h>
#include <spice/cuda/util/event.h>
#include <spice/cuda/util/stream.h>
#include <spice/snn.h>
#include <spice/util/span2d.h>

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
	multi_snn( spice::util::layout desc, float dt, int_ delay = 1, bool bench = false );
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
		spice::util::span2d<uint_> counts; // gpu x iter

		std::vector<int_ *> ddata_data;
		spice::util::span2d<int_ *> ddata;

		std::vector<uint_ *> dcounts_data;
		spice::util::span2d<uint_ *> dcounts;
	} _spikes;

	std::array<std::optional<util::stream>, util::device::max_devices> _cp;

	std::array<std::thread, util::device::max_devices> _workers;
	std::atomic_bool _running{ true };
	std::atomic_int32_t _iter{ 0 };
	std::atomic_int32_t _work{ 0 };
	std::vector<int_> * _out_spikes = nullptr;
	std::mutex _out_spikes_lock;
	std::atomic_bool _bench{ false };
	std::array<size_, util::device::max_devices> _slices;
	std::array<double, util::device::max_devices> _timings;

	std::vector<int_> _tmp;

	multi_snn( float dt, int_ delay );
};
} // namespace spice::cuda