#pragma once

#include <spice/snn_info.h>
#include <spice/util/adj_list.h>
#include <spice/util/span2d.h>

#include <cuda_runtime.h>


namespace spice
{
namespace cuda
{
void generate_rnd_adj_list( spice::util::adj_list::int4 const * layout, int len, int * out_edges );

template <typename Model>
void upload_storage(
    typename Model::neuron::ptuple_t const & neuron,
    typename Model::synapse::ptuple_t const & synapse );

template <typename Model>
void init( snn_info info, spice::util::span2d<int const> adj = {} );

template <typename Model>
void update(
    snn_info info,
    float dt,
    int * spikes,
    unsigned * out_num_spikes,

    spice::util::span2d<unsigned> history = {},
    int * ages = nullptr,
    int * updates = nullptr,
    unsigned * num_updates = nullptr,
    int const iter = 0,
    int const delay = 0,
    int const max_history = 0,
    spice::util::span2d<int const> adj = {} );

template <typename Model>
void receive(
    snn_info info,
    spice::util::span2d<int const> adj,

    int const * spikes,
    unsigned const * num_spikes,

    int * ages = nullptr,
    spice::util::span2d<unsigned> history = {},
    int max_history = 0,
    int iter = 0,
    int delay = 0,
    float dt = 0 );

template <typename T>
void zero_async( T * t, cudaStream_t s = nullptr );
} // namespace cuda
} // namespace spice
