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
void generate_rnd_adj_list( cudaStream_t s, spice::util::layout const & desc, int_ * edges );

template <typename Model>
void upload_meta(
    cudaStream_t s,
    typename Model::neuron::ptuple_t const & neuron,
    typename Model::synapse::ptuple_t const & synapse );

template <typename Model>
void init(
    cudaStream_t s,
    int_ slice_width,
    int_ n,
    int_ i,
    snn_info info,
    spice::util::span2d<int_ const> adj = {} );

template <typename Model>
void update(
    cudaStream_t s,

    int_ slice_width,
    int_ n,
    int_ i,
    snn_info info,
    float dt,
    int_ * spikes,
    uint_ * out_num_spikes,

    uint_ * history = nullptr,
    int_ * ages = nullptr,
    int_ * updates = nullptr,
    uint_ * num_updates = nullptr,
    int_ const iter = 0,
    int_ const delay = 0 );

template <typename Model>
void receive(
    cudaStream_t s,

    snn_info info,
    spice::util::span2d<int_ const> adj,

    int_ const * spikes,
    uint_ const * num_spikes,
    int_ const * updates,
    uint_ const * num_updates,

    int_ * ages = nullptr,
    uint_ * history = nullptr,
    int_ iter = 0,
    float dt = 0 );

template <typename T>
void zero_async( T * t, cudaStream_t s = nullptr );
} // namespace cuda
} // namespace spice
