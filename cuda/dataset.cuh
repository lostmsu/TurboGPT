#pragma once
#include "types.cuh"
#include "../utils.h"

// One endpoint calculation per sequence, shared by all of its byte positions.
template <int Context>
__device__ __forceinline__ void load_tile(const DeviceState &d, BlockWorkspace &s, int tile,
                                          uint64_t step) {
    for (int local = threadIdx.x; local < TileTokens / Context; local += BlockThreads) {
        int sequence = tile * (TileTokens / Context) + local;
        int64_t endpoint = 0;
        if (sequence < d.config.batch && !d.input)
            endpoint =
                8 + int64_t(hash_u64(d.config.seed + step * uint64_t(d.config.batch) + sequence) %
                            uint64_t(d.size - 8));
#pragma unroll
        for (int pos = 0; pos < Context; ++pos) {
            int row = local * Context + pos, token = sequence * Context + pos;
            if (sequence >= d.config.batch) {
                s.x[row] = s.y[row] = 0;
            } else if (d.input) {
                s.x[row] = d.input[token];
                s.y[row] = d.target[token];
            } else {
                s.x[row] = d.data[endpoint - Context + pos];
                s.y[row] = d.data[endpoint - Context + pos + 1];
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileTokens * 16; i += BlockThreads)
        s.hidden[i] = d.weight_high[d.layout.embedding + s.x[i / 16] * 16 + i % 16];
    __syncthreads();
}
