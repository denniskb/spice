#pragma once

#include <spice/cuda/snn.h>

#include <vector>

namespace spice::cuda
{
template <typename Model>
class multi_snn
{
public:
	multi_snn( spice::util::layout desc, float dt, int delay = 1 );

	void step();
	void sync();

private:
	std::vector<snn<Model>> _nets;
	float const _dt;
	int const _delay;

	static spice::util::layout
	slice( spice::util::layout const & whole, std::size_t n, std::size_t i );
};
} // namespace spice::cuda