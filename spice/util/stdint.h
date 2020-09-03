#pragma once

#include <cstddef>
#include <cstdint>


using int_ = std::int32_t;
using uint_ = std::uint32_t;
using long_ = std::int64_t;
using ulong_ = std::uint64_t;
using size_ = std::size_t;

inline constexpr size_ operator"" _sz( unsigned long long n ) { return n; }
