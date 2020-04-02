#pragma once


#ifdef __CUDACC__
#define if_constexpr if
#else
#define if_constexpr if constexpr
#endif