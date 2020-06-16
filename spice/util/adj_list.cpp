#include <spice/cuda/util/defs.h>
#include <spice/util/adj_list.h>
#include <spice/util/assert.h>
#include <spice/util/random.h>
#include <spice/util/type_traits.h>

#include <ctime>
#include <numeric>
#include <random>


static int _seed = 1339;


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
std::size_t adj_list::generate_layout( neuron_group const & desc, std::vector<int> & layout )
{
	// #neurons * #edges(desc) degrees
	layout.resize( desc.size() * desc.connections().size() );
	std::vector<int> degrees_acc( desc.size() );

	xorwow gen( _seed++ );

	{
		int * d = layout.data();
		for( auto edge : desc.connections() )
		{
			util::binomial_distribution<> binom(
			    narrow_int<int>( desc.size( std::get<1>( edge ) ) ), std::get<2>( edge ) );

			for( std::size_t i = 0, n = desc.first( std::get<0>( edge ) ); i < n; i++ ) d[i] = 0;

			for( std::size_t i = desc.first( std::get<0>( edge ) ),
			                 n = desc.last( std::get<0>( edge ) );
			     i < n;
			     i++ )
			{
				int const x = binom( gen );
				d[i] = x;
				degrees_acc[i] += x;
			}

			for( std::size_t i = desc.last( std::get<0>( edge ) ), n = desc.size(); i < n; i++ )
				d[i] = 0;

			d += desc.size();
		}
	}

	return ( *std::max_element( degrees_acc.begin(), degrees_acc.end() ) + WARP_SZ - 1 ) / WARP_SZ *
	       WARP_SZ;
} // namespace spice::util

// static
std::size_t adj_list::generate( neuron_group const & desc, std::vector<int> & edges )
{
	std::vector<int> layout;

	auto const degree = generate_layout( desc, layout );
	edges.resize( degree * desc.size() );
	generate( desc, layout, edges );

	return degree;
}

int const * adj_list::edges() const { return _edges; }

std::size_t adj_list::num_nodes() const { return _num_nodes; }
std::size_t adj_list::max_degree() const { return _max_degree; }
std::size_t adj_list::num_edges() const { return num_nodes() * max_degree(); }


// static
void adj_list::generate(
    neuron_group const & desc, std::vector<int> const & layout, std::vector<int> & edges )
{
	xorwow gen( _seed++ );
	std::exponential_distribution<float> exp;
	auto bounded_exp = [&]( auto & gen ) {
		return std::min( -std::log( 1.0f / gen.max() ), exp( gen ) );
	};

	std::size_t const N = desc.size();
	std::size_t const max_d = edges.size() / N;

	std::size_t offset = 0;
	for( std::size_t i = 0; i < N; i++ )
	{
		for( std::size_t j = 0; j < desc.connections().size(); j++ )
		{
			int const degree = layout[i + j * N];

			if( !degree ) continue;

			int const first =
			    narrow_int<int>( desc.first( std::get<1>( desc.connections().at( j ) ) ) );
			int const range =
			    narrow_int<int>( desc.range( std::get<1>( desc.connections().at( j ) ) ) );

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

		while( offset < ( i + 1 ) * max_d ) edges[offset++] = -1;
	}
}
} // namespace spice::util
