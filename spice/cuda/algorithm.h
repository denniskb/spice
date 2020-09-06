#pragma once

#include <spice/snn_info.h>
#include <spice/util/adj_list.h>
#include <spice/util/layout.h>
#include <spice/util/span2d.h>

#include <cuda_runtime.h>


namespace spice
{
namespace cuda
{
void generate_rnd_adj_list( spice::util::layout const & desc, int_ * edges );

template <typename Model>
void upload_meta(
    typename Model::neuron::ptuple_t const & neuron,
    typename Model::synapse::ptuple_t const & synapse );

template <typename Model>
void init( int_ first, int_ last, snn_info info, spice::util::span2d<int_ const> adj = {} );

template <typename Model>
void update(
    cudaStream_t s,

    int_ first,
    int_ last,
    snn_info info,
    float dt,
    int_ * spikes,
    uint_ * out_num_spikes,

    spice::util::span2d<uint_> history = {},
    int_ * ages = nullptr,
    int_ * updates = nullptr,
    uint_ * num_updates = nullptr,
    int_ const iter = 0,
    int_ const delay = 0,
    int_ const max_history = 0,
    spice::util::span2d<int_ const> adj = {} );

template <typename Model>
void receive(
    cudaStream_t s,

    snn_info info,
    spice::util::span2d<int_ const> adj,

    int_ const * spikes,
    uint_ const * num_spikes,

    int_ * ages = nullptr,
    spice::util::span2d<uint_> history = {},
    int_ max_history = 0,
    int_ iter = 0,
    int_ delay = 0,
    float dt = 0 );

template <typename T>
void zero_async( T * t, cudaStream_t s = nullptr );
} // namespace cuda
} // namespace spice
