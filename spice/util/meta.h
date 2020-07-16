#pragma once

#include <spice/util/host_defines.h>
#include <spice/util/if_constexpr.h>

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
template <typename Tuple, typename Fn, std::size_t... I>
void for_each_i( Tuple & t, Fn && f, std::index_sequence<I...> )
{
	auto const tmp = std::make_tuple( ( std::forward<Fn>( f )( std::get<I>( t ), I ), 0 )... );
}

template <typename Tuple, typename Fn, std::size_t... I>
auto map( Tuple & t, Fn && f, std::index_sequence<I...> )
{
	return std::make_tuple( std::forward<Fn>( f )( std::get<I>( t ) )... );
}

template <typename TupleDst, typename TupleSrc, typename Fn, std::size_t... I>
void map( TupleDst & tdst, TupleSrc const & tsrc, Fn && f, std::index_sequence<I...> )
{
	auto const tmp = std::make_tuple(
	    ( std::forward<Fn>( f )( std::get<I>( tdst ), std::get<I>( tsrc ) ), 0 )... );
}
} // namespace detail

template <typename Tuple, typename Fn>
void for_each_i( Tuple & t, Fn && f )
{
	detail::for_each_i(
	    t, std::forward<Fn>( f ), std::make_index_sequence<std::tuple_size_v<Tuple>>() );
}

template <typename Tuple, typename Fn>
void for_each( Tuple & t, Fn && f )
{
	for_each_i( t, [&f]( auto & elem, int ) { std::forward<Fn>( f )( elem ); } );
}

template <typename Tuple, typename Fn>
auto map( Tuple & t, Fn && f )
{
	return detail::map(
	    t, std::forward<Fn>( f ), std::make_index_sequence<std::tuple_size_v<Tuple>>() );
}

template <typename TupleDst, typename TupleSrc, typename Fn>
void map( TupleDst & tdst, TupleSrc const & tsrc, Fn && f )
{
	static_assert(
	    std::tuple_size_v<TupleDst> == std::tuple_size_v<TupleSrc>,
	    "both tuples must have the same size" );

	return detail::map(
	    tdst,
	    tsrc,
	    std::forward<Fn>( f ),
	    std::make_index_sequence<std::tuple_size_v<TupleDst>>() );
}

// TODO: Rewrite without constepxr
template <typename Tuple, typename T, typename TransOp, typename ReduceOp, int I = 0>
T transform_reduce(
    Tuple const & t1, Tuple const & t2, T init, TransOp && trans_op, ReduceOp && reduce_op )
{
	if_constexpr( I == std::tuple_size_v<Tuple> ) return init;
	else return std::forward<ReduceOp>( reduce_op )(
	    std::forward<TransOp>( trans_op )( std::get<I>( t1 ), std::get<I>( t2 ) ),
	    transform_reduce<Tuple, T, TransOp, ReduceOp, I + 1>(
	        t1,
	        t2,
	        init,
	        std::forward<TransOp>( trans_op ),
	        std::forward<ReduceOp>( reduce_op ) ) );
}


template <typename... T>
struct type_list
{
	using tuple_t = std::tuple<T...>;
	using ptuple_t = std::tuple<T *...>;
	using cptuple_t = std::tuple<const T *...>;
	static constexpr std::size_t size = sizeof...( T );
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

template <
    template <typename> typename Cont,
    typename TypeList,
    typename Base = decltype( make_soa<Cont>( TypeList() ) )>
struct soa_t : public Base
{
	using Base::Base;

	explicit soa_t( std::size_t n )
	{
		for_each( *this, [n]( auto & cont ) { cont.resize( n ); } );
	}

	void resize( std::size_t n )
	{
		for_each( *this, [n]( auto & cont ) { cont.resize( n ); } );
	}

	typename TypeList::ptuple_t data()
	{
		return map( *this, []( auto & cont ) { return cont.data(); } );
	}

	typename TypeList::cptuple_t data() const
	{
		return map( *this, []( auto & cont ) { return cont.data(); } );
	}

	template <std::size_t sz = TypeList::size, std::enable_if_t<sz != 0> * = nullptr>
	std::size_t size() const
	{
		return std::get<0>( *this ).size();
	}

	template <std::size_t sz = TypeList::size, std::enable_if_t<sz == 0, int> * = nullptr>
	std::size_t size() const
	{
		return 0;
	}

	std::vector<typename TypeList::tuple_t> to_aos() const
	{
		std::vector<typename TypeList::tuple_t> result( size() );
		_to_aos( result );
		return result;
	}

	void from_aos( std::vector<typename TypeList::tuple_t> const & src ) { _from_aos( src ); }

private:
	// Recursive helper for 'to_aos()'.
	// Only defined if tuple size > 0
	template <
	    std::size_t I = 0,
	    std::size_t sz = TypeList::size,
	    std::enable_if_t<sz != 0> * = nullptr>
	void _to_aos( std::vector<typename TypeList::tuple_t> & out ) const
	{
		std::vector<typename std::tuple_element<I, typename TypeList::tuple_t>::type> tmp =
		    std::get<I>( *this );
		for( std::size_t i = 0; i < out.size(); i++ ) std::get<I>( out[i] ) = tmp[i];

		_to_aos<I + 1>( out );
	}

	// Specialization of (above). Terminates recursion
	template <>
	void _to_aos<TypeList::size, TypeList::size, nullptr>(
	    std::vector<typename TypeList::tuple_t> & ) const
	{
	}

	// Noop-version of '_to_aos()' in case of tuple size == 0.
	template <
	    std::size_t I = 0,
	    std::size_t sz = TypeList::size,
	    std::enable_if_t<sz == 0, int> * = nullptr>
	void _to_aos( std::vector<typename TypeList::tuple_t> & ) const
	{
	}


	// Recursive helper for 'to_aos()'.
	// Only defined if tuple size > 0
	template <
	    std::size_t I = 0,
	    std::size_t sz = TypeList::size,
	    std::enable_if_t<sz != 0> * = nullptr>
	void _from_aos( std::vector<typename TypeList::tuple_t> const & src )
	{
		std::vector<std::tuple_element_t<I, typename TypeList::tuple_t>> tmp( src.size() );
		for( std::size_t i = 0; i < tmp.size(); i++ ) tmp[i] = std::get<I>( src[i] );

		std::get<I>( *this ) = tmp;

		_from_aos<I + 1>( src );
	}

	// Specialization of (above). Terminates recursion
	template <>
	void _from_aos<TypeList::size, TypeList::size, nullptr>(
	    std::vector<typename TypeList::tuple_t> const & )
	{
	}

	// Noop-version of '_from_aos()' in case of tuple size == 0.
	template <
	    std::size_t I = 0,
	    std::size_t sz = TypeList::size,
	    std::enable_if_t<sz == 0, int> * = nullptr>
	void _from_aos( std::vector<typename TypeList::tuple_t> const & )
	{
	}
};
} // namespace util
} // namespace spice

namespace std
{
template <template <typename> typename Cont, typename TypeList>
struct tuple_size<spice::util::soa_t<Cont, TypeList>>
{
	static constexpr std::size_t value = TypeList::size;
};
} // namespace std
#pragma warning( pop )