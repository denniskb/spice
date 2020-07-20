#pragma once

#include <spice/util/layout.h>
#include <spice/util/span.hpp>

#include <vector>


namespace spice
{
namespace util
{
// view
class adj_list
{
public:
	adj_list() = default;
	adj_list( std::size_t num_nodes, std::size_t max_degree, int const * edges );

	nonstd::span<int const> neighbors( std::size_t i_node ) const;
	std::size_t edge_index( std::size_t i_src, std::size_t i_dst ) const;

	static void generate( layout const & desc, std::vector<int> & edges );

	int const * edges() const;

	std::size_t num_nodes() const;
	std::size_t max_degree() const;
	std::size_t num_edges() const;

private:
	std::size_t _num_nodes = 0;
	std::size_t _max_degree = 0;
	int const * _edges = nullptr;
};
} // namespace util
} // namespace spice
