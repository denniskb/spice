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
	spice_assert( num_nodes < std::numeric_limits<int>::max(), "invalid node count" );
	spice_assert( max_degree < std::numeric_limits<int>::max(), "invalid degree" );
	spice_assert(
	    num_nodes * max_degree < std::numeric_limits<int>::max(), "no. of edges out of boudns" );
}


nonstd::span<int const> adj_list::neighbors( std::size_t i_node ) const
{
	spice_assert( i_node < num_nodes(), "index out of bounds" );

	auto const first = &_edges[i_node * max_degree()];

	std::ptrdiff_t d = narrow_int<ptrdiff_t>( max_degree() ) - 1;
	while( d >= 0 && first[d] < 0 ) --d;

	return {first, static_cast<std::size_t>( d + 1 )};
}

int adj_list::edge_index( std::size_t i_src, std::size_t i_dst ) const
{
	spice_assert( i_src < num_nodes(), "index out of bounds" );
	spice_assert( i_dst < neighbors( i_src ).size(), "index out of bounds" );

	return narrow_cast<int>( i_src * max_degree() + i_dst );
}


// static
std::size_t adj_list::generate( neuron_group const & desc, std::vector<int4> & layout )
{
	layout.clear();
	layout.reserve( 2 * desc.size() );

	std::vector offsets( desc.size(), 0 );

	xorwow gen( _seed++ );

	for( auto edge : desc.connections() )
	{
		util::binomial_distribution<> binom(
		    narrow_int<int>( desc.size( std::get<1>( edge ) ) ), std::get<2>( edge ) );

		std::pair<int, int> const bound(
		    narrow_cast<int>( desc.first( std::get<1>( edge ) ) ),
		    narrow_cast<int>( desc.range( std::get<1>( edge ) ) ) );

		for( std::size_t i = desc.first( std::get<0>( edge ) ),
		                 n = desc.last( std::get<0>( edge ) );
		     i < n;
		     i++ )
		{
			auto const degree = binom( gen );

			offsets[i] += degree;
			layout.push_back( {-1, degree, bound.first, bound.second} );
		}
	}

	auto const max_degree =
	    ( *std::max_element( offsets.begin(), offsets.end() ) + WARP_SZ - 1 ) / WARP_SZ * WARP_SZ;

	for( std::size_t i = 0; i < offsets.size(); i++ )
	{
		int const offset = narrow_cast<int>( i * max_degree );

		layout.push_back( {offset + offsets[i], max_degree - offsets[i], -1, -1} );
		offsets[i] = offset;
	}

	std::size_t j = 0;
	for( auto edge : desc.connections() )
	{
		for( std::size_t i = desc.first( std::get<0>( edge ) ),
		                 n = desc.last( std::get<0>( edge ) );
		     i < n;
		     i++, j++ )
		{
			layout[j].offset = offsets[i];
			offsets[i] += layout[j].degree; // this is a (per-thread) register in CUDA
		}
	}

	return max_degree;
} // namespace spice::util

// static
std::size_t adj_list::generate( neuron_group const & desc, std::vector<int> & edges )
{
	std::vector<int4> layout;

	auto const degree = generate( desc, layout );
	edges.resize( degree * desc.size() );
	generate( layout, edges );

	return degree;
}

int const * adj_list::edges() const { return _edges; }

std::size_t adj_list::num_nodes() const { return _num_nodes; }
std::size_t adj_list::max_degree() const { return _max_degree; }
std::size_t adj_list::num_edges() const { return num_nodes() * max_degree(); }


// static
void adj_list::generate( std::vector<int4> const & layout, std::vector<int> & edges )
{
	xorwow gen( _seed++ );
	std::exponential_distribution<float> exp;
	auto bounded_exp = [&]( auto & gen ) {
		return std::min( -std::log( 1.0f / gen.max() ), exp( gen ) );
	};

	for( std::size_t i = 0; i < layout.size(); i++ )
	{
		auto const desc = layout[i];

		if( desc.first == -1 )
		{
			for( std::size_t j = 0; j < desc.degree; j++ ) edges.at( desc.offset + j ) = -1;

			continue;
		}

		float * neighbor_ids = reinterpret_cast<float *>( edges.data() + desc.offset );

		float total = bounded_exp( gen );
		for( std::size_t j = 0; j < desc.degree; j++ )
		{
			neighbor_ids[j] = total;
			total += bounded_exp( gen );
		}

		float const scale = ( desc.range - desc.degree ) / total;
		for( int j = 0; j < desc.degree; j++ )
			edges.at( desc.offset + j ) =
			    desc.first + narrow_cast<int>( neighbor_ids[j] * scale ) + j;
	}
}
} // namespace spice::util
