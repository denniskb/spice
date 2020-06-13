#include <spice/util/neuron_group.h>

#include <spice/util/assert.h>
#include <spice/util/type_traits.h>

#include <numeric>


namespace spice::util
{
neuron_group::neuron_group( std::size_t const num_neurons, float const connectivity )
    : neuron_group( {num_neurons}, {{0, 0, connectivity}} )
{
}

#pragma warning( push )
#pragma warning( disable : 4189 ) // unreferenced variable 'gs' in assert
neuron_group::neuron_group(
    std::vector<std::size_t> const & group_sizes, std::vector<edge> const & connectivity )
    : _group_sizes( group_sizes )
    , _connectivity( connectivity )
{
	for( auto gs : group_sizes )
		spice_assert( gs <= std::numeric_limits<int>::max(), "invalid group size" );

	for( auto c : connectivity )
	{
		spice_assert(
		    std::get<0>( c ) < group_sizes.size() && std::get<1>( c ) < group_sizes.size(),
		    "invalid index in connectivity matrix" );
		spice_assert(
		    std::get<2>( c ) >= 0.0f && std::get<2>( c ) <= 1.0f, "invalid connect. prob." );
	}

	std::sort( _connectivity.begin(), _connectivity.end(), cmp );

	edge e( -1, -1, 0.0f );
	for( auto c : _connectivity )
	{
		spice_assert(
		    std::get<0>( c ) != std::get<0>( e ) || std::get<1>( c ) != std::get<1>( e ),
		    "no duplicate edges in connectivity list allowed" );
		e = c;
	}
}
#pragma warning( pop )


std::size_t neuron_group::num_groups() const
{
	return narrow_int<std::size_t>( _group_sizes.size() );
}

std::size_t neuron_group::size() const
{
	return std::accumulate( _group_sizes.begin(), _group_sizes.end(), std::size_t( 0 ) );
}
std::size_t neuron_group::size( std::size_t i ) const
{
	spice_assert( i < num_groups(), "index out of range" );
	return _group_sizes[i];
}

std::size_t neuron_group::first( std::size_t i ) const
{
	spice_assert( i < num_groups(), "index out of range" );
	return std::accumulate( _group_sizes.begin(), _group_sizes.begin() + i, std::size_t( 0 ) );
}
std::size_t neuron_group::last( std::size_t i ) const
{
	spice_assert( i < num_groups(), "index out of range" );
	return std::accumulate( _group_sizes.begin(), _group_sizes.begin() + i + 1, std::size_t( 0 ) );
}
std::size_t neuron_group::range( std::size_t i ) const
{
	spice_assert( i < num_groups(), "index out of range" );
	return last( i ) - first( i );
}

nonstd::span<neuron_group::edge const> neuron_group::connections() const { return _connectivity; }
nonstd::span<neuron_group::edge const> neuron_group::neighbors( std::size_t const i ) const
{
	auto const first =
	    std::lower_bound( _connectivity.begin(), _connectivity.end(), edge( i, 0, 0.0f ), cmp );

	auto const last = std::upper_bound(
	    _connectivity.begin(),
	    _connectivity.end(),
	    edge( i, std::numeric_limits<std::size_t>::max(), 0.0f ),
	    cmp );

	return {_connectivity.data() + ( first - _connectivity.begin() ),
	        static_cast<std::size_t>( last - first )};
}


// static
bool neuron_group::cmp( edge const & a, edge const & b )
{
	return std::get<0>( a ) < std::get<0>( b ) ||
	       std::get<0>( a ) == std::get<0>( b ) && std::get<1>( a ) < std::get<1>( b );
}
} // namespace spice::util
