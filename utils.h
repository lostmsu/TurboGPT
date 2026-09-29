#pragma once
#include <chrono>
#include <cstdint>

using Clock = std::chrono::steady_clock;
inline double seconds(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now() - t).count();
}

#ifdef __CUDACC__
__host__ __device__
#endif
    inline uint64_t hash_u64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}
