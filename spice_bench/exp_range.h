#pragma once

#include <benchmark/benchmark.h>

inline void
exp_range( benchmark::internal::Benchmark * b, int64_t start, int64_t stop, double mul = 2 )
{
	for( ; start <= stop; start = ( int64_t )( start * mul ) ) b->Args( { start } );
}

inline void exp_ranges(
    benchmark::internal::Benchmark * b, std::vector<int64_t> range1, std::vector<int64_t> range2 )
{
	auto mul1 = range1.size() > 2 ? range1[2] : 2;
	auto mul2 = range2.size() > 2 ? range2[2] : 2;

	for( ; range1[0] <= range1[1]; range1[0] *= mul1 )
		for( int64_t i = range2[0]; i <= range2[1]; i *= mul2 ) b->Args( { range1[0], i } );
}

// clang-format off
#define R( ... ) {__VA_ARGS__}
// clang-format on

#define ExpRange( start, stop, ... )                        \
	Apply( []( benchmark::internal::Benchmark * b ) {       \
		exp_range( b, ( start ), ( stop ), ##__VA_ARGS__ ); \
	} )

#define ExpRanges( X, Y ) \
	Apply( []( benchmark::internal::Benchmark * b ) { exp_ranges( b, X, Y ); } )
