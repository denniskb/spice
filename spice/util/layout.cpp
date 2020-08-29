#include <spice/util/layout.h>

#include <spice/cuda/util/defs.h>
#include <spice/util/assert.h>
#include <spice/util/type_traits.h>

#include <cmath>
#include <map>
#include <numeric>


static constexpr std::size_t operator"" _sz( unsigned long long int n ) { return n; }

static std::size_t estimate_max_deg( std::vector<spice::util::layout::edge> const & connections )
{
	using namespace spice::util;

	int src = connections.empty() ? 0 : std::get<0>( connections.front() );

	std::size_t result = 0;
	double m = 0.0, s2 = 0.0;
	for( auto c : connections )
	{
		if( std::get<0>( c ) != src )
		{
			result = std::max( result, narrow_cast<std::size_t>( m + 3 * std::sqrt( s2 ) ) );
			src = std::get<0>( c );
			m = 0.0;
			s2 = 0.0;
		}

		auto const dst_range = static_cast<double>( std::get<3>( c ) - std::get<2>( c ) );
		m += dst_range * std::get<4>( c );
		s2 += dst_range * std::get<4>( c ) * ( 1.0 - std::get<4>( c ) );
	}

	return ( std::max( result, narrow_cast<std::size_t>( m + 3 * std::sqrt( s2 ) ) ) + WARP_SZ -
	         1 ) /
	       WARP_SZ * WARP_SZ;
}

static std::vector<std::tuple<std::size_t, std::size_t, float>> p2connect( float p )
{
	if( p )
		return { { 0, 0, p } };
	else
		return {};
}


namespace spice::util
{
layout::layout( std::size_t const num_neurons, float const connections )
    : layout( { num_neurons }, p2connect( connections ) )
{
	spice_assert( num_neurons > 0, "layout must contain at least 1 neuron" );
}

#pragma warning( push )
#pragma warning( disable : 4189 4457 ) // unreferenced variable 'gs' in assert, hidden variable
layout::layout(
    std::vector<std::size_t> const & pops,
    std::vector<std::tuple<std::size_t, std::size_t, float>> connections )
{
	spice_assert( pops.size() > 0, "layout must contain at least 1 (non-empty) population" );

	// Validate
	for( auto pop : pops )
		spice_assert(
		    pop > 0 && pop <= std::numeric_limits<int>::max(), "invalid population size" );

	for( auto c : connections )
	{
		spice_assert(
		    std::get<0>( c ) < pops.size() && std::get<1>( c ) < pops.size(),
		    "invalid index in connections matrix" );
		spice_assert(
		    std::get<2>( c ) > 0.0f && std::get<2>( c ) <= 1.0f, "invalid connect. prob." );
	}

	{
		std::sort( connections.begin(), connections.end(), []( auto const & a, auto const & b ) {
			return std::get<0>( a ) < std::get<0>( b ) ||
			       ( std::get<0>( a ) == std::get<0>( b ) && std::get<1>( a ) < std::get<1>( b ) );
		} );

		for( std::size_t i = 1; i < connections.size(); i++ )
		{
			spice_assert(
			    std::get<0>( connections[i] ) != std::get<0>( connections[i - 1] ) ||
			        std::get<1>( connections[i] ) != std::get<1>( connections[i - 1] ),
			    "no duplicate edges in connections list allowed" );
		}

		// Initialize
		auto const first = [&]( std::size_t i ) {
			spice_assert( i < pops.size(), "index out of range" );
			return std::accumulate( pops.begin(), pops.begin() + i, 0_sz );
		};
		auto const last = [&]( std::size_t i ) {
			spice_assert( i < pops.size(), "index out of range" );
			return std::accumulate( pops.begin(), pops.begin() + i + 1, 0_sz );
		};

		_n = std::accumulate( pops.begin(), pops.end(), 0_sz );

		for( auto c : connections )
		{
			_connections.push_back(
			    { narrow<int>( first( std::get<0>( c ) ) ),
			      narrow<int>( last( std::get<0>( c ) ) ),
			      narrow<int>( first( std::get<1>( c ) ) ),
			      narrow<int>( last( std::get<1>( c ) ) ),
			      std::get<2>( c ) } );
		}

		_max_degree = estimate_max_deg( _connections );
	}

	spice_assert( max_degree() % WARP_SZ == 0 );
}
#pragma warning( pop )

std::size_t layout::size() const { return _n; }
std::vector<layout::edge> const & layout::connections() const { return _connections; }
std::size_t layout::max_degree() const { return _max_degree; }

layout::slice<> layout::cut( std::size_t n, std::size_t i )
{
	spice_assert( n > 0 );
	spice_assert( i < n );

	std::vector<int> szs;
	std::vector<int> costs;
	{
		std::map<int, std::pair<int, double>> pop2sizedeg;
		for( auto const & c : connections() )
		{
			pop2sizedeg[std::get<0>( c )].first = std::get<1>( c ) - std::get<0>( c );

			auto x = pop2sizedeg[std::get<2>( c )];
			x.first = std::get<3>( c ) - std::get<2>( c );
			x.second += ( std::get<1>( c ) - std::get<0>( c ) ) * std::get<4>( c );
			pop2sizedeg[std::get<2>( c )] = x;
		}

		for( auto const & [k, v] : pop2sizedeg )
		{
			szs.push_back( v.first );
			costs.push_back( static_cast<int>( std::round( v.first * v.second ) ) );
		}
	}
	std::inclusive_scan( szs.begin(), szs.end(), szs.begin() );
	std::inclusive_scan( costs.begin(), costs.end(), costs.begin() );

	auto partition = [&]( std::size_t pivot ) -> std::size_t {
		if( !pivot ) return 0;
		auto I = std::lower_bound( costs.begin(), costs.end(), pivot );
		auto i = I - costs.begin();

		if( i ) pivot -= costs[i - 1];

		return pivot * ( szs[i] - ( i ? szs[i - 1] : 0 ) ) /
		           ( costs[i] - ( i ? costs[i - 1] : 0 ) ) +
		       ( i ? szs[i - 1] : 0 );
	};

	int const first = narrow<int>( partition( costs.back() * i / n ) );
	int const last = narrow<int>( partition( costs.back() * ( i + 1 ) / n ) );

	std::vector<layout::edge> part;
	for( auto c : connections() )
	{
		std::get<2>( c ) = std::max( first, std::get<2>( c ) );
		std::get<3>( c ) = std::min( last, std::get<3>( c ) );
		if( std::get<2>( c ) < std::get<3>( c ) ) part.push_back( std::move( c ) );
	}

	return { layout( size(), part ), first, last };
}

layout::layout( std::size_t n, std::vector<edge> flat )
    : _n( n )
    , _connections( flat )
    , _max_degree( estimate_max_deg( flat ) )
{
}
} // namespace spice::util
