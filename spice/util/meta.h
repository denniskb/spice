#pragma once

#include <spice/util/host_defines.h>

#include <tuple>
#include <type_traits>
#include <utility>


#pragma warning( push )
#pragma warning( disable : 4100 ) // unreferenced parameters on last loop iteration and empty tuples
namespace spice
{
namespace util
{
namespace detail
{
template <std::size_t I>
struct integral_constant
{
	constexpr operator std::size_t() const { return I; }
};

template <typename Tuple, typename Fn, std::size_t... I>
auto map_i( Tuple && t, Fn && f, std::index_sequence<I...> )
{
	return std::make_tuple( std::forward<Fn>( f )(
	    std::get<I>( std::forward<Tuple>( t ) ), integral_constant<I>() )... );
}

template <typename Tuple1, typename Tuple2, typename Fn, std::size_t... I>
auto map_i( Tuple1 && t1, Tuple2 && t2, Fn && f, std::index_sequence<I...> )
{
	static_assert(
	    std::tuple_size_v<std::decay_t<Tuple1>> == std::tuple_size_v<std::decay_t<Tuple2>>,
	    "tuples must have the same size" );

	return std::make_tuple( std::forward<Fn>( f )(
	    std::get<I>( std::forward<Tuple1>( t1 ) ),
	    std::get<I>( std::forward<Tuple2>( t2 ) ),
	    integral_constant<I>() )... );
}
} // namespace detail

template <typename Tuple, typename Fn>
auto map_i( Tuple && t, Fn && f )
{
	return detail::map_i(
	    std::forward<Tuple>( t ),
	    std::forward<Fn>( f ),
	    std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple>>>() );
}

template <typename Tuple1, typename Tuple2, typename Fn>
auto map_i( Tuple1 && t1, Tuple2 && t2, Fn && f )
{
	return detail::map_i(
	    std::forward<Tuple1>( t1 ),
	    std::forward<Tuple2>( t2 ),
	    std::forward<Fn>( f ),
	    std::make_index_sequence<std::tuple_size_v<std::decay_t<Tuple1>>>() );
}

template <typename Tuple, typename Fn>
auto map( Tuple && t, Fn && f )
{
	return map_i( std::forward<Tuple>( t ), [&f]( auto && x, auto ) {
		return std::forward<Fn>( f )( std::forward<decltype( x )>( x ) );
	} );
}

template <typename Tuple1, typename Tuple2, typename Fn>
auto map( Tuple1 && t1, Tuple2 && t2, Fn && f )
{
	return map_i(
	    std::forward<Tuple1>( t1 ), std::forward<Tuple2>( t2 ), [&f]( auto && x, auto && y, auto ) {
		    return std::forward<Fn>( f )(
		        std::forward<decltype( x )>( x ), std::forward<decltype( x )>( x ) );
	    } );
}

template <typename Tuple, typename Fn>
void for_each_i( Tuple & t, Fn && f )
{
	map_i( t, [&f]( auto && elem, auto I ) {
		std::forward<Fn>( f )( std::forward<decltype( elem )>( elem ), I );
		return 0;
	} );
}

template <typename Tuple, typename Fn>
void for_each( Tuple & t, Fn && f )
{
	for_each_i( t, [&]( auto && elem, auto ) {
		std::forward<Fn>( f )( std::forward<decltype( elem )>( elem ) );
	} );
}

template <typename Tuple, typename T, typename Fn, std::size_t I = 0>
auto reduce( Tuple && t, T && init, Fn && f )
{
	if constexpr( I == std::tuple_size_v<std::decay_t<Tuple>> )
		return init;
	else
		return std::forward<Fn>( f )(
		    std::get<I>( std::forward<Tuple>( t ) ),
		    reduce<Tuple, T, Fn, I + 1>(
		        std::forward<Tuple>( t ), std::forward<T>( init ), std::forward<Fn>( f ) ) );
}


template <typename... T>
struct type_list
{
	using tuple_t = std::tuple<T...>;
	using ptuple_t = std::tuple<T *...>;
	using cptuple_t = std::tuple<const T *...>;
	static constexpr std::size_t size = sizeof...( T );

	template <template <typename> typename Cont>
	using soa_t = std::tuple<Cont<T>...>;
};

template <int I, typename Iter>
HYBRID auto & get( Iter it )
{
	return it.template get<I>();
}


template <template <typename> typename Cont, typename... Fields>
auto make_soa( type_list<Fields...> )
{
	return std::tuple<Cont<Fields>...>();
}

template <template <typename> typename Cont, typename TypeList>
struct soa_t
{
	soa_t() = default;

	explicit soa_t( std::size_t n )
	{
		for_each( _data, [n]( auto & cont ) { cont.resize( n ); } );
	}

	void resize( std::size_t n )
	{
		for_each( _data, [n]( auto & cont ) { cont.resize( n ); } );
	}

	typename TypeList::ptuple_t data()
	{
		return map( _data, []( auto & cont ) { return cont.data(); } );
	}

	typename TypeList::cptuple_t data() const
	{
		return map( _data, []( auto & cont ) { return cont.data(); } );
	}

	template <std::size_t sz = TypeList::size, std::enable_if_t<sz != 0> * = nullptr>
	std::size_t size() const
	{
		return std::get<0>( _data ).size();
	}

	template <std::size_t sz = TypeList::size, std::enable_if_t<sz == 0, int> * = nullptr>
	std::size_t size() const
	{
		return 0;
	}

	std::vector<typename TypeList::tuple_t> to_aos() const
	{
		std::vector<typename TypeList::tuple_t> result( size() );

		for_each_i( _data, [&]( auto const & cont, auto I ) {
			std::vector<typename std::tuple_element<I, typename TypeList::tuple_t>::type> tmp =
			    cont;
			for( std::size_t i = 0; i < result.size(); i++ ) std::get<I>( result[i] ) = tmp[i];
		} );

		return result;
	}

	void from_aos( std::vector<typename TypeList::tuple_t> const & src )
	{
		for_each_i( _data, [&]( auto & cont, auto I ) {
			std::vector<std::tuple_element_t<I, typename TypeList::tuple_t>> tmp( src.size() );
			for( std::size_t i = 0; i < tmp.size(); i++ ) tmp[i] = std::get<I>( src[i] );

			cont = tmp;
		} );
	}

private:
	typename TypeList::template soa_t<Cont> _data;
};
} // namespace util
} // namespace spice
#pragma warning( pop )