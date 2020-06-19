#include <spice/cuda/util/defs.h>
#include <spice/util/adj_list.h>
#include <spice/util/assert.h>
#include <spice/util/random.h>
#include <spice/util/type_traits.h>

#include <ctime>
#include <numeric>
#include <random>


static unsigned long long _seed = 1337;


namespace spice::util
{
adj_list::adj_list( std::size_t num_nodes, std::size_t max_degree, int const * edges )
    : _num_nodes( num_nodes )
    , _max_degree( max_degree )
    , _edges( edges )
{
	spice_assert( num_nodes <= std::numeric_limits<int>::max(), "invalid node count" );
	spice_assert( max_degree <= std::numeric_limits<int>::max(), "invalid degree" );
	spice_assert(
	    num_nodes * max_degree <= std::numeric_limits<unsigned>::max(),
	    "no. of edges out of boudns" );
}


nonstd::span<int const> adj_list::neighbors( std::size_t i_node ) const
{
	spice_assert( i_node < num_nodes(), "index out of bounds" );

	auto const first = &_edges[i_node * max_degree()];

	std::ptrdiff_t d = narrow_int<ptrdiff_t>( max_degree() ) - 1;
	while( d >= 0 && first[d] < 0 ) --d;

	return {first, static_cast<std::size_t>( d + 1 )};
}

std::size_t adj_list::edge_index( std::size_t i_src, std::size_t i_dst ) const
{
	spice_assert( i_src < num_nodes(), "index out of bounds" );
	spice_assert( i_dst < neighbors( i_src ).size(), "index out of bounds" );

	return i_src * max_degree() + i_dst;
}

// static
void adj_list::generate( neuron_group & desc, std::vector<int> & edges )
{
	edges.resize( desc.size() * desc.max_degree() );

	xorshift64 gen( _seed++ );
	std::exponential_distribution<float> exp;
	auto bounded_exp = [&]( auto & gen ) {
		return std::min( -std::log( 1.0f / gen.max() ), exp( gen ) );
	};

	std::size_t const N = desc.size();

	std::size_t offset = 0;
	for( std::size_t i = 0; i < N; i++ )
	{
		int total_degree = 0;
		for( auto const c : desc.connections() )
		{
			if( i < desc.first( std::get<0>( c ) ) || i >= desc.last( std::get<0>( c ) ) ) continue;

			util::binomial_distribution<> binom(
			    narrow_int<int>( desc.size( std::get<1>( c ) ) ), std::get<2>( c ) );

			int const degree =
			    std::min( narrow_int<int>( desc.max_degree() - total_degree ), binom( gen ) );
			total_degree += degree;

			int const first = narrow_int<int>( desc.first( std::get<1>( c ) ) );
			int const range = narrow_int<int>( desc.range( std::get<1>( c ) ) );

			float * neighbor_ids = reinterpret_cast<float *>( edges.data() + offset );

			float total = bounded_exp( gen );
			for( std::size_t k = 0; k < degree; k++ )
			{
				neighbor_ids[k] = total;
				total += bounded_exp( gen );
			}

			float const scale = ( range - degree ) / total;
			for( int k = 0; k < degree; k++ )
				edges[offset++] = first + narrow_cast<int>( neighbor_ids[k] * scale ) + k;
		}

		while( offset < ( i + 1 ) * desc.max_degree() ) edges[offset++] = -1;
	}
}

int const * adj_list::edges() const { return _edges; }

std::size_t adj_list::num_nodes() const { return _num_nodes; }
std::size_t adj_list::max_degree() const { return _max_degree; }
std::size_t adj_list::num_edges() const { return num_nodes() * max_degree(); }
} // namespace spice::util
