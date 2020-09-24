#include <spice/util/layout.h>

#include <spice/cuda/util/defs.h>
#include <spice/util/assert.h>
#include <spice/util/stdint.h>
#include <spice/util/type_traits.h>

#include <cmath>
#include <map>
#include <numeric>


static size_ estimate_max_deg( std::vector<spice::util::layout::edge> const & connections )
{
	using namespace spice::util;

	int_ src = connections.empty() ? 0 : std::get<0>( connections.front() );

	size_ result = 0;
	double m = 0.0, s2 = 0.0;
	for( auto c : connections )
	{
		if( std::get<0>( c ) != src )
		{
			result = std::max( result, narrow_cast<size_>( m + 3 * std::sqrt( s2 ) ) );
			src = std::get<0>( c );
			m = 0.0;
			s2 = 0.0;
		}

		auto const dst_range = static_cast<double>( std::get<3>( c ) - std::get<2>( c ) );
		m += dst_range * std::get<4>( c );
		s2 += dst_range * std::get<4>( c ) * ( 1.0 - std::get<4>( c ) );
	}

	return ( std::max( result, narrow_cast<size_>( m + 3 * std::sqrt( s2 ) ) ) + WARP_SZ - 1 ) /
	       WARP_SZ * WARP_SZ;
}


namespace spice::util
{
layout::layout( size_ const num_neurons, float const connections )
    : layout( { num_neurons }, { { 0, 0, connections } } )
{
	spice_assert( num_neurons > 0, "layout must contain at least 1 neuron" );
}

#pragma warning( push )
#pragma warning( disable : 4189 4457 ) // unreferenced variable 'gs' in assert, hidden variable
layout::layout(
    std::vector<size_> const & pops, std::vector<std::tuple<size_, size_, float>> connections )
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
		    std::get<2>( c ) >= 0.0f && std::get<2>( c ) <= 1.0f, "invalid connect. prob." );
	}

	{
		std::sort( connections.begin(), connections.end(), []( auto const & a, auto const & b ) {
			return std::get<0>( a ) < std::get<0>( b ) ||
			       ( std::get<0>( a ) == std::get<0>( b ) && std::get<1>( a ) < std::get<1>( b ) );
		} );

		for( size_ i = 1; i < connections.size(); i++ )
		{
			spice_assert(
			    std::get<0>( connections[i] ) != std::get<0>( connections[i - 1] ) ||
			        std::get<1>( connections[i] ) != std::get<1>( connections[i - 1] ),
			    "no duplicate edges in connections list allowed" );
		}

		// Initialize
		auto const first = [&]( size_ i ) {
			spice_assert( i < pops.size(), "index out of range" );
			return std::accumulate( pops.begin(), pops.begin() + i, 0_sz );
		};
		auto const last = [&]( size_ i ) {
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

size_ layout::size() const { return _n; }
std::vector<layout::edge> const & layout::connections() const { return _connections; }
size_ layout::max_degree() const { return _max_degree; }

std::pair<size_, size_> layout::static_load_balance( size_ const n, size_ const i ) const
{
	spice_assert( n > 0 );
	spice_assert( i < n );

	std::vector<int> szs;
	std::vector<size_> costs;
	{
		std::map<int, std::pair<int, double>> pop2size_cost;
		for( auto const & c : connections() )
		{
			size_ const src_size = std::get<1>( c ) - std::get<0>( c );
			size_ const dst_size = std::get<3>( c ) - std::get<2>( c );

			pop2size_cost[std::get<0>( c )].first = src_size;

			auto x = pop2size_cost[std::get<2>( c )];
			x.first = dst_size;
			x.second += src_size * dst_size * static_cast<double>( std::get<4>( c ) );
			pop2size_cost[std::get<2>( c )] = x;
		}

		for( auto const & [k, v] : pop2size_cost )
		{
			szs.push_back( v.first );
			costs.push_back( v.first + static_cast<size_>( std::round( v.second ) ) );
		}
	}
	std::inclusive_scan( szs.begin(), szs.end(), szs.begin() );
	std::inclusive_scan( costs.begin(), costs.end(), costs.begin() );

	auto partition = [&]( size_ pivot ) -> size_ {
		if( !pivot ) return 0;
		auto I = std::lower_bound( costs.begin(), costs.end(), pivot );
		auto i = I - costs.begin();

		if( i ) pivot -= costs[i - 1];

		return pivot * ( szs[i] - ( i ? szs[i - 1] : 0 ) ) /
		           ( costs[i] - ( i ? costs[i - 1] : 0 ) ) +
		       ( i ? szs[i - 1] : 0 );
	};

	return { partition( costs.back() * i / n ), partition( costs.back() * ( i + 1 ) / n ) };
}

layout::slice<> layout::cut( std::pair<size_, size_> range ) const
{
	spice_assert( range.first <= size() );
	spice_assert( range.second <= size() );
	spice_assert( range.first <= range.second );

	std::vector<layout::edge> part;
	for( auto c : connections() )
	{
		std::get<2>( c ) = std::max( narrow<int_>( range.first ), std::get<2>( c ) );
		std::get<3>( c ) = std::min( narrow<int_>( range.second ), std::get<3>( c ) );
		if( std::get<2>( c ) < std::get<3>( c ) ) part.push_back( std::move( c ) );
	}

	return { layout( size(), part ), range.first, range.second };
}

layout layout::cut( size_ slice_width, size_ n_gpus, size_ i_gpu ) const
{
	spice_assert( slice_width > 0 );
	spice_assert( i_gpu < n_gpus );

	std::vector<layout::edge> part;
	for( auto c : connections() )
	{
		for( size_ first = i_gpu * slice_width; first < size(); first += n_gpus * slice_width )
		{
			size_ last = first + slice_width;
			auto const a = std::max( narrow<int_>( first ), std::get<2>( c ) );
			auto const b = std::min( narrow<int_>( last ), std::get<3>( c ) );
			if( a < b )
				part.push_back( { std::get<0>( c ), std::get<1>( c ), a, b, std::get<4>( c ) } );
		}
	}

	return { size(), part };
}

layout::layout( size_ n, std::vector<edge> flat )
    : _n( n )
    , _connections( flat )
    , _max_degree( estimate_max_deg( flat ) )
{
}
} // namespace spice::util
