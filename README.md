![banner by Hans Braxmeier](spices.jpg)

# Spice

Spice (/spaɪk/) is a multi-GPU, time-driven (aka clock-based), general-purpose spiking neural network simulator. Its features include:
- State of the art performance (incl. sub-second setup)
- Multi-GPU support with linear scaling up to 8 GPUs
- Ability to define custom models in native C++
- Modern, user-friendly API

## Publications
- [Even Faster SNN Simulation with Lazy+Event-driven Plasticity and Shared Atomics](https://bautembach.de/#hpec2021) (2021)
- [Multi-GPU SNN Simulation with Perfect Static Load Balancing](https://bautembach.de/#ijcnn2021) (2021) (branch:[gather](https://github.com/denniskb/spice/tree/gather))
- [Faster and Simpler SNN Simulation with Work Queues](https://bautembach.de/#ijcnn2020) (2020)

## Requirements
- CUDA SDK 11 or higher
- nVidia GPU with compute capability 6.1 (10XX or newer)
- CMAKE 3.0.2 or higher, recommended default settings, builds out of the box.
- C++17 compatible compiler
- (Only tested on Linux: Spice contains no Linux-specific code and *should* build on Windows, but hasn't been tested on it for some time.)

## Usage
The "samples" project currently implements benchmarks used for our publications. It compiles to an executable that can be invoked via
```
./samples --model {vogels|brunel|brunel+} --nsyn 1234 --gpu {single|multi} --bench {sim|setup}
```
which prints a json object similar to
```
{
  "sim": "samples",
  "model": "brunel",
  "#syn": 1000000000,
  "#gpus": 1,
  "simtime": 1.49136,
  "setuptime": 0.081864
}
```
The field "simtime" is the ratio between simulation time (wall clock time) and biological time. All benchmarks simulate 10s of biological time. "setuptime" is the absolute setup time in seconds.

## Defining Custom Models
Have a look at the sample models defined in [`spice/models`](https://github.com/denniskb/spice/tree/master/spice/models); the syntax is pretty sraight-forward. Currently, the easiest way to define your own models is to hack one of the existing ones. To instantiate your model you'd write
```
cuda::snn<mymodel> net(
  {10, 20, 30},   // create three neuron populations with 10, 20, 30 neurons respectively
  {{0, 2, 0.1},   // randomly connect pop. A->C with prob. 10%
   {1, 2, 0.05}}, // randomly connect pop. B->C with prob. 5%
  0.001,          // time step in seconds
  8               // synaptic delay
);

while (true)
  net.step();
```
`cuda::snn` can be any of {`cpu::snn`|`cuda::snn`|`cuda::multi_snn`} depending on whether you want to run on a (single core) CPU, single GPU, or multiple GPUs. Beware that `net.step()` executes `delay` many steps when `net` is of type `cuda::multi_snn` in order to be able to hide latency from spike synchronization. `net.step()` also takes an optional pointer to a `std::vector<int>` and writes spiking data into it.
