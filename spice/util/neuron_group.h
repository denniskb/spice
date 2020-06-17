#pragma once

#include <spice/util/span.hpp>

#include <tuple>
#include <vector>


namespace spice
{
namespace util
{
class neuron_group
{
public:
	// (src_id, dst_id, p_connect)
	using edge = std::tuple<std::size_t, std::size_t, float>;

	neuron_group( std::size_t num_neurons, float connectivity );
	neuron_group(
	    std::vector<std::size_t> const & group_sizes, std::vector<edge> const & connectivity );

	std::size_t num_groups() const;
	std::size_t size() const;
	std::size_t size( std::size_t i ) const;
	std::size_t first( std::size_t i ) const;
	std::size_t last( std::size_t i ) const;
	std::size_t range( std::size_t i ) const;

	nonstd::span<edge const> connections() const;
	nonstd::span<edge const> neighbors( std::size_t i ) const;

	std::size_t max_degree();

private:
	std::vector<std::size_t> _group_sizes;
	std::vector<edge> _connectivity;
	std::size_t _max_degree;

	static bool cmp( edge const & a, edge const & b );
};
} // namespace util
} // namespace spice
