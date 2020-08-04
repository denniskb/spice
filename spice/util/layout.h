#pragma once

#include <spice/util/span.hpp>

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
	template <typename T>
	friend class ::spice::cuda::multi_snn;

public:
	// ([first1, last2), [first2, last2), p)
	using edge = std::tuple<int, int, int, int, float>;

	layout( std::size_t num_neurons, float connections );
	layout(
	    std::vector<std::size_t> const & group_sizes,
	    std::vector<std::tuple<std::size_t, std::size_t, float>> connectivity );

	std::size_t size() const;
	nonstd::span<edge const> connections() const;
	std::size_t max_degree() const;

	layout slice( std::size_t n, std::size_t i );

private:
	std::size_t _n;
	std::vector<edge> _connections;
	std::size_t _max_degree;

	layout( std::size_t n, std::vector<edge> flat );
};
} // namespace util
} // namespace spice
