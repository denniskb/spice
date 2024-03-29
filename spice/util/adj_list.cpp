#include <spice/cuda/util/defs.h>
#include <spice/util/adj_list.h>
#include <spice/util/assert.h>
#include <spice/util/random.h>
#include <spice/util/type_traits.h>

#include <ctime>
#include <numeric>


static ulong_ _seed = 1337;


namespace spice::util
{
adj_list::adj_list( size_ num_nodes, size_ max_degree, int_ const * edges )
    : _num_nodes( num_nodes )
    , _max_degree( max_degree )
    , _edges( edges )
{
	spice_assert( num_nodes <= std::numeric_limits<int>::max(), "invalid node count" );
	spice_assert( max_degree <= std::numeric_limits<int>::max(), "invalid degree" );
	spice_assert(
	    num_nodes * max_degree <= std::numeric_limits<uint_>::max(), "no. of edges out of boudns" );
}


nonstd::span<int_ const> adj_list::neighbors( size_ i_node ) const
{
	spice_assert( i_node < num_nodes(), "index out of bounds" );

	auto const first = &_edges[i_node * max_degree()];
	return {
	    first,
	    narrow_cast<size_>(
	        std::lower_bound( first, first + max_degree(), std::numeric_limits<int_>::max() ) -
	        first ) };
}

size_ adj_list::edge_index( size_ i_src, size_ i_dst ) const
{
	spice_assert( i_src < num_nodes(), "index out of bounds" );
	spice_assert( i_dst < neighbors( i_src ).size(), "index out of bounds" );

	return i_src * max_degree() + i_dst;
}

// static
void adj_list::generate( layout const & desc, std::vector<int> & edges )
{
	edges.resize( desc.size() * desc.max_degree() );

	xoroshiro256ss gen( _seed++ );

	int_ const N = narrow<int>( desc.size() );

	size_ offset = 0;
	for( int_ i = 0; i < N; i++ )
	{
		int_ total_degree = 0;
		for( auto const c : desc.connections() )
		{
			if( i < std::get<0>( c ) || i >= std::get<1>( c ) ) continue;

			int_ const first = std::get<2>( c );
			int_ const range = std::get<3>( c ) - first;

			int_ const degree = std::min(
			    narrow<int>( desc.max_degree() - total_degree ),
			    binornd( gen, range, std::get<4>( c ) ) );

			total_degree += degree;

			float * neighbor_ids = reinterpret_cast<float *>( edges.data() + offset );

			float total = exprnd( gen );
			for( int_ k = 0; k < degree; k++ )
			{
				neighbor_ids[k] = total;
				total += exprnd( gen );
			}

			float const scale = ( range - degree ) / total;
			for( int_ k = 0; k < degree; k++ )
				edges[offset++] = first + narrow_cast<int>( neighbor_ids[k] * scale ) + k;
		}

		while( offset < ( i + 1 ) * desc.max_degree() )
			edges[offset++] = std::numeric_limits<int_>::max();
	}
}

int_ const * adj_list::edges() const { return _edges; }

size_ adj_list::num_nodes() const { return _num_nodes; }
size_ adj_list::max_degree() const { return _max_degree; }
size_ adj_list::num_edges() const { return num_nodes() * max_degree(); }
} // namespace spice::util
