#pragma once
#include "types.cuh"

// Causal running-max attention: retain only the winning key and its GELU gate.
template <int Context>
__device__ __forceinline__ int selected_key(const bf16 *qkv, int row, int head, float &top) {
    int start = row / Context * Context, best = start;
    top = -CUDART_INF_F;
    for (int key = start; key <= row; ++key) {
        float score = 0;
        for (int c = 0; c < 4; ++c)
            score += float(qkv[row * 48 + head * 4 + c]) * float(qkv[key * 48 + 16 + head * 4 + c]);
        if (score > top) {
            top = score;
            best = key;
        } // first index wins ties
    }
    return best;
}
template <int Context>
__device__ __forceinline__ void attention_forward(LayerActivations &a, bf16 *out, bool reuse) {
    for (int i = threadIdx.x; i < TileTokens * 4; i += BlockThreads) {
        int row = i / 4, head = i % 4;
        int key;
        float g;
        if (reuse) {
            key = a.selected[i];
            g = a.gates[i];
        } else {
            float top;
            key = selected_key<Context>(a.qkv, row, head, top);
            float e = 1.f + erff(top * 0.7071067811865475f);
            g = 0.5f * top * e;
            // Preserve only the winning key and its gate, never a T-by-T matrix.
            a.selected[i] = key;
            a.gates[i] = g;
            a.derivatives[i] = 0.5f * e + top * 0.3989422804014327f * expf(-0.5f * top * top);
        }
        for (int col = head * 4; col < head * 4 + 4; ++col)
            out[row * 16 + col] = __float2bfloat16_rn(float(a.qkv[key * 48 + 32 + col]) * g);
    }
    __syncthreads();
}
template <int Context>
__device__ __forceinline__ void attention_backward(BlockWorkspace &s, const LayerActivations &a,
                                                   const bf16 *dy, bf16 *dqkv) {
    // Forward cached the winner and GELU terms. Only its upstream gradient
    // depends on dy; compute it once for each query/head.
    for (int i = threadIdx.x; i < TileTokens * 4; i += BlockThreads) {
        int query = i / 4, head = i % 4;
        int key = a.selected[i];
        float dt = 0;
        for (int c = 0; c < 4; ++c)
            dt += float(dy[query * 16 + head * 4 + c]) * float(a.qkv[key * 48 + 32 + head * 4 + c]);
        s.result[i] = dt * a.derivatives[i];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < TileTokens * 48; i += BlockThreads) {
        int row = i / 48, kind = i % 48 / 16, col = i % 16, head = col / 4;
        float value = 0;
        int first = kind == 0 ? row : row / Context * Context;
        int end = kind == 0 ? row + 1 : first + Context;
        for (int query = first; query < end; ++query) {
            int key = a.selected[query * 4 + head];
            if (kind != 0 && key != row)
                continue;
            float dt = s.result[query * 4 + head];
            if (kind == 0)
                value = dt * float(a.qkv[key * 48 + 16 + col]);
            else if (kind == 1)
                value += dt * float(a.qkv[query * 48 + col]);
            else
                value += float(dy[query * 16 + col]) * a.gates[query * 4 + head];
        }
        dqkv[i] = __float2bfloat16_rn(value);
    }
    __syncthreads();
}
