#pragma once
#include "ops.cuh"
#include "attention.cuh"
#include "weights.cuh"

// GPT forward and backward, including the output head and cross-entropy loss.
template <int Context>
__device__ __forceinline__ void forward(const DeviceState &d, BlockWorkspace &s,
                                        LayerActivations *saved, int tile) {
    begin_weights(d, s);
    for (int layer = 0; layer < d.config.depth; ++layer) {
        LayerActivations &a = saved[layer];
        const LayerParameters &p = d.layout.layers[layer];
        LayerWeights w = layer_weights(d, s, layer, false);
        for (int i = threadIdx.x; i < TileTokens * 16; i += BlockThreads)
            a.x[i] = s.hidden[i];
        normalize(d, a.x, p.attention_norm, s.normalized);
        project<16, 48>(s.normalized, w.qkv, a.qkv);
        rope<Context>(a.qkv);
        attention_forward<Context>(a, s.branch, false);
        project<16, 16>(s.branch, w.attention, a.middle, a.x);
        normalize<CacheWeights>(d, a.middle, p.mlp_norm, s.normalized);
        swiglu(s, w.gate);
        if (d.mlp_cache) {
            bf16 *cache = d.mlp_cache + (blockIdx.x * d.config.depth + layer) * TileTokens * 96;
            for (int i = threadIdx.x; i < TileTokens * 96; i += BlockThreads)
                cache[i] = s.wide[i];
        }
        project<48, 16>(s.work, w.down, s.hidden, a.middle);
    }
    normalize(d, s.hidden, d.layout.final_norm, s.normalized);
    project<16, 256, HeadStride>(d, s.normalized, d.layout.head, s.wide);
    // One warp per token: stable FP32 softmax, then BF16 dlogits. Logits are
    // never materialized in global memory except in explicit diagnostic calls.
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int row = warp; row < TileTokens; row += BlockWarps) {
        bool valid = tile * TileTokens + row < d.config.batch * Context;
        float high = -CUDART_INF_F;
        for (int col = lane; col < 256; col += 32)
            high = fmaxf(high, float(s.wide[row * HeadStride + col]));
        for (int k = 16; k; k >>= 1)
            high = fmaxf(high, __shfl_xor_sync(0xffffffff, high, k));
        float sum = 0;
        for (int col = lane; col < 256; col += 32)
            sum += expf(float(s.wide[row * HeadStride + col]) - high);
        sum = warp_sum(sum);
        float loss = high + logf(sum) - float(s.wide[row * HeadStride + s.y[row]]);
        if (lane == 0 && valid)
            d.token_losses[tile * TileTokens + row] = loss;
        for (int col = lane; col < 256; col += 32) {
            float logit = float(s.wide[row * HeadStride + col]);
            if (d.logits && valid)
                d.logits[(tile * TileTokens + row) * 256 + col] = logit;
            float g = (expf(logit - high) / sum - (col == s.y[row])) / (d.config.batch * Context);
            s.wide[row * HeadStride + col] = __float2bfloat16_rn(valid ? g : 0.f);
        }
    }
    __syncthreads();
}
template <int Context>
__device__ __forceinline__ void backward(const DeviceState &d, BlockWorkspace &s,
                                         LayerActivations *saved, float *grad, bool first) {
    weight_gradient(s.normalized, s.wide, 16, 256, HeadStride, grad + d.layout.head, first);
    matmul(s, s.wide, HeadStride, 1, d.weight_high + d.layout.head, 16, 1, TileTokens, 16, 256);
    cast_result(s, s.branch, TileTokens * 16);
    norm_backward(d, s.hidden, s.branch, d.layout.final_norm, s.dhidden, grad, false);
    for (int layer = d.config.depth - 1; layer >= 0; --layer) {
        LayerActivations &a = saved[layer];
        const LayerParameters &p = d.layout.layers[layer];
        LayerWeights w = layer_weights(d, s, layer, true);
        // On large-L2 devices, reuse gate/up outputs saved by forward.
        normalize<CacheWeights>(d, a.middle, p.mlp_norm, s.normalized);
        if (d.mlp_cache) {
            const bf16 *cache =
                d.mlp_cache + (blockIdx.x * d.config.depth + layer) * TileTokens * 96;
            for (int i = threadIdx.x; i < TileTokens * 96; i += BlockThreads)
                s.wide[i] = cache[i];
            __syncthreads();
            swiglu_activation(s);
        } else {
            swiglu(s, w.gate);
        }
        linear_backward(d, s, s.work, s.dhidden, p.down, 48, 16, s.work, grad, first, w.down);
        swiglu_backward(s);
        linear_backward(d, s, s.normalized, s.wide, p.gate, 16, 96, s.branch, grad, first, w.gate);
        norm_backward(d, a.middle, s.branch, p.mlp_norm, s.dhidden, grad, true);
        attention_forward<Context>(a, s.work, true);
        linear_backward(d, s, s.work, s.dhidden, p.attention, 16, 16, s.branch, grad, first,
                        w.attention);
        attention_backward<Context>(s, a, s.branch, s.dqkv);
        rope<Context, true>(s.dqkv);
        normalize(d, a.x, p.attention_norm, s.normalized);
        linear_backward(d, s, s.normalized, s.dqkv, p.qkv, 16, 48, s.branch, grad, first, w.qkv);
        norm_backward(d, a.x, s.branch, p.attention_norm, s.dhidden, grad, true);
    }
    for (int i = threadIdx.x; i < TileTokens * 16; i += BlockThreads)
        atomicAdd(grad + d.layout.embedding + s.x[i / 16] * 16 + i % 16, float(s.dhidden[i]));
    __syncthreads();
}
