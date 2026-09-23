#pragma once
#include "types.cuh"
#include <cuda_pipeline.h>

// The logits workspace has spare space during transformer layers. Reuse it
// for one MLP weight buffer and two alternating QKV/output buffers, without
// increasing shared memory or reducing block residency.
constexpr bool CacheWeights = TileTokens == 32;
struct LayerWeights {
    const bf16 *qkv, *attention, *gate, *down;
};
__device__ __forceinline__ void copy_weights(bf16 *target, const bf16 *source, int count) {
    for (int i = threadIdx.x * 8; i < count; i += BlockThreads * 8)
        __pipeline_memcpy_async(target + i, source + i, 16);
}
__device__ __forceinline__ bf16 *weight_buffer(BlockWorkspace &s) {
    return s.wide + TileTokens * 96;
}
__device__ __forceinline__ void begin_weights(const DeviceState &d, BlockWorkspace &s) {
    if constexpr (CacheWeights) {
        copy_weights(weight_buffer(s), d.weight_high + d.layout.layers[0].qkv, 1024);
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
    }
}
__device__ __forceinline__ LayerWeights layer_weights(const DeviceState &d, BlockWorkspace &s,
                                                      int layer, bool backward) {
    const auto &p = d.layout.layers[layer];
    if constexpr (CacheWeights) {
        bf16 *base = weight_buffer(s), *qkv = base + (layer % 2) * 1024, *mlp = base + 2048;
        copy_weights(mlp, d.weight_high + p.gate, 2304);
        if (backward) {
            copy_weights(qkv, d.weight_high + p.qkv, 1024);
        } else if (layer + 1 < d.config.depth) {
            copy_weights(base + ((layer + 1) % 2) * 1024,
                         d.weight_high + d.layout.layers[layer + 1].qkv, 1024);
        }
        __pipeline_commit();
        return {qkv, qkv + 768, mlp, mlp + 1536};
    }
    return {d.weight_high + p.qkv, d.weight_high + p.attention, d.weight_high + p.gate,
            d.weight_high + p.down};
}
