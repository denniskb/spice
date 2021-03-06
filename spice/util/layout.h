#pragma once

#include <spice/util/span.hpp>
#include <spice/util/stdint.h>

#include <tuple>
#include <vector>


namespace spice
{
namespace cuda
{
template <typename T>
class multi_snn;
} // namespace cuda
} // namespace spice


namespace spice
{
namespace util
{
class layout
{
public:
	// ([first1, last2), [first2, last2), p)
	using edge = std::tuple<int_, int_, int_, int_, float>;

	layout( size_ num_neurons, float connections );
	layout(
	    std::vector<size_> const & group_sizes,
	    std::vector<std::tuple<size_, size_, float>> connectivity );

	size_ size() const;
	std::vector<edge> const & connections() const;
	size_ max_degree() const;

	std::pair<size_, size_> static_load_balance( size_ n, size_ i ) const;
	template <typename T = layout>
	struct slice
	{
		T part;
		size_ first;
		size_ last;
	};
	slice<> cut( std::pair<size_, size_> range ) const;

	layout cut( size_ slice_width, size_ n_gpus, size_ i_gpu ) const;

private:
	size_ _n;
	std::vector<edge> _connections;
	size_ _max_degree;

	layout( size_ n, std::vector<edge> flat );
};
} // namespace util
} // namespace spice
