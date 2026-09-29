#pragma once
#include "types.cuh"

// Partial RoPE, as in GLM-5.3 and MiMo-V2.6-Pro: each 4-wide query/key head rotates its
// first (interleaved) pair by one radian per position; the second pair has no position.
// With a single rotated pair the frequency is theta^0, so no base theta applies.
static __constant__ float RopeCos[8] = {1.0f, 0.5403022766113281f, -0.416146844625473f,
                                        -0.9899924993515015f, -0.6536436080932617f,
                                        0.28366219997406006f, 0.9601702690124512f,
                                        0.7539022564888f};
static __constant__ float RopeSin[8] = {0.0f, 0.8414709568023682f, 0.9092974066734314f,
                                        0.14112000167369843f, -0.756802499294281f,
                                        -0.9589242935180664f, -0.279415488243103f,
                                        0.6569865942001343f};

// Rotates query and key heads in place after the QKV projection. Inverse applies the
// transpose, turning rotated-space query/key gradients into projection gradients.
// Unfused products keep the rounding identical to the PyTorch reference.
template <int Context, bool Inverse = false>
__device__ __forceinline__ void rope(bf16 *qkv) {
    for (int i = threadIdx.x; i < TileTokens * 8; i += BlockThreads) {
        int row = i / 8, pos = row % Context;
        bf16 *pair = qkv + row * 48 + i % 8 * 4; // query heads 0-3, then key heads 0-3
        float x = float(pair[0]), y = float(pair[1]);
        float cos = RopeCos[pos], sin = Inverse ? -RopeSin[pos] : RopeSin[pos];
        pair[0] = __float2bfloat16_rn(__fsub_rn(__fmul_rn(x, cos), __fmul_rn(y, sin)));
        pair[1] = __float2bfloat16_rn(__fadd_rn(__fmul_rn(x, sin), __fmul_rn(y, cos)));
    }
    __syncthreads();
}

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
