#pragma once

#include <spice/util/layout.h>
#include <spice/util/span.hpp>
#include <spice/util/stdint.h>

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
	adj_list( size_ num_nodes, size_ max_degree, int_ const * edges );

	nonstd::span<int_ const> neighbors( size_ i_node ) const;
	size_ edge_index( size_ i_src, size_ i_dst ) const;

	static void generate( layout const & desc, std::vector<int> & edges );

	int_ const * edges() const;

	size_ num_nodes() const;
	size_ max_degree() const;
	size_ num_edges() const;

private:
	size_ _num_nodes = 0;
	size_ _max_degree = 0;
	int_ const * _edges = nullptr;
};
} // namespace util
} // namespace spice
