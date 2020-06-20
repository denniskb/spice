#pragma once

#ifdef __CUDA_ARCH__
#define HYBRID __device__
#else
#define HYBRID
#endif