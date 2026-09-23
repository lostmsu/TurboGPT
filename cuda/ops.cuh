#pragma once
#include "mma.cuh"
#include <cuda_pipeline.h>

// BF16/FP32 storage, tensor-core products, RMSNorm and SwiGLU.
__device__ __forceinline__ float precise_weight(const DeviceState &d, int i) {
    return __uint_as_float((unsigned(__bfloat16_as_ushort(d.weight_high[i])) << 16) |
                           d.weight_low[i]);
}
__device__ __forceinline__ void write_weight(const DeviceState &d, int i, float value) {
    unsigned bits = __float_as_uint(value);
    d.weight_high[i] = __ushort_as_bfloat16(bits >> 16);
    d.weight_low[i] = uint16_t(bits);
}
__device__ __forceinline__ float warp_sum(float x) {
    for (int k = 16; k; k >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, k);
    return x;
}
__device__ __forceinline__ float row_sum(float x) {
    for (int k = 8; k; k >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, k);
    return x;
}
__device__ __forceinline__ float sigmoid(float x) {
    return 1.f / (1.f + expf(-x));
}

// Arbitrary row-major matrices or their transposes, in 16x8 warp tiles.
__device__ __forceinline__ void matmul(BlockWorkspace &s, const bf16 *a, int ar, int ac,
                                       const bf16 *b, int br, int bc, int m, int n, int k) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int tile = warp; tile < (m / 16) * (n / 8); tile += BlockWarps) {
        int row = tile / (n / 8) * 16, col = tile % (n / 8) * 8;
        float accum[4] = {};
        for (int inner = 0; inner < k; inner += 16) {
            unsigned af[4], bf[2];
            load_a(af, a, ar, ac, row, inner);
            load_b(bf, b, br, bc, inner, col);
            mma_16x8(accum, af, bf);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int index = (row + lane / 4 + half * 8) * n + col + (lane % 4) * 2;
            *reinterpret_cast<float2 *>(s.result + index) =
                make_float2(accum[half * 2], accum[half * 2 + 1]);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void cast_result(BlockWorkspace &s, bf16 *out, int size) {
    for (int i = threadIdx.x; i < size; i += BlockThreads)
        out[i] = __float2bfloat16_rn(s.result[i]);
    __syncthreads();
}
template <bool WaitForWeights = false>
__device__ __forceinline__ void normalize(const DeviceState &d, const bf16 *x, int norm, bf16 *y) {
    for (int i = threadIdx.x; i < TileTokens * 16; i += BlockThreads) {
        float v = float(x[i]);
        float inv = rsqrtf(row_sum(v * v) / 16 + 1e-6f);
        y[i] = __float2bfloat16_rn(v * inv * precise_weight(d, norm + i % 16));
    }
    if constexpr (WaitForWeights)
        __pipeline_wait_prior(0);
    __syncthreads();
}
__device__ __forceinline__ void norm_backward(const DeviceState &d, const bf16 *x, const bf16 *dy,
                                              int norm, bf16 *dx, float *grad, bool residual) {
    for (int i = threadIdx.x; i < TileTokens * 16; i += BlockThreads) {
        float v = float(x[i]), g = float(dy[i]);
        float inv = rsqrtf(row_sum(v * v) / 16 + 1e-6f), z = v * inv;
        float dz = g * precise_weight(d, norm + i % 16);
        float value = (dz - z * row_sum(dz * z) / 16) * inv;
        // Native graph keeps residual gradients in BF16 between operations.
        bf16 rounded = __float2bfloat16_rn(value);
        dx[i] = residual ? __float2bfloat16_rn(float(dx[i]) + float(rounded)) : rounded;
        atomicAdd(grad + norm + i % 16, g * z);
    }
    __syncthreads();
}
// Store matrix gradients straight from accumulator fragments. A shared FP32
// result, a scalar copy, and their extra block barrier are unnecessary here.
__device__ __forceinline__ void weight_gradient(const bf16 *x, const bf16 *dy, int in, int out,
                                                int dy_stride, float *gradient, bool first) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int tile = warp; tile < (out / 16) * (in / 8); tile += BlockWarps) {
        int row = tile / (in / 8) * 16, col = tile % (in / 8) * 8;
        float accum[4] = {};
#pragma unroll
        for (int inner = 0; inner < TileTokens; inner += 16) {
            unsigned af[4], bf[2];
            load_a(af, dy, 1, dy_stride, row, inner);
            load_b(bf, x, in, 1, inner, col);
            mma_16x8(accum, af, bf);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int index = (row + lane / 4 + half * 8) * in + col + (lane % 4) * 2;
            float2 value = make_float2(accum[half * 2], accum[half * 2 + 1]);
            if (!first) {
                float2 previous = *reinterpret_cast<const float2 *>(gradient + index);
                value.x += previous.x;
                value.y += previous.y;
            }
            *reinterpret_cast<float2 *>(gradient + index) = value;
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void linear_backward(const DeviceState &d, BlockWorkspace &s,
                                                const bf16 *x, const bf16 *dy, int weight, int in,
                                                int out, bf16 *dx, float *grad, bool first,
                                                const bf16 *weights = nullptr) {
    weight_gradient(x, dy, in, out, out, grad + weight, first);
    matmul(s, dy, out, 1, weights ? weights : d.weight_high + weight, in, 1, TileTokens, in, out);
    cast_result(s, dx, TileTokens * in);
}
// The gate and up matrices are contiguous, so one GEMM computes both.
// Preserve BF16 rounding at SiLU and multiply, matching PyTorch's SwiGLU.
__device__ __forceinline__ void swiglu_activation(BlockWorkspace &s) {
    for (int i = threadIdx.x; i < TileTokens * 48; i += BlockThreads) {
        int j = (i / 48) * 96 + i % 48;
        float g = float(s.wide[j]), u = float(s.wide[j + 48]);
        bf16 activated = __float2bfloat16_rn(g * sigmoid(g));
        s.work[i] = __float2bfloat16_rn(float(activated) * u);
    }
    __syncthreads();
}
__device__ __forceinline__ void swiglu(BlockWorkspace &s, const bf16 *gate) {
    project<16, 96>(s.normalized, gate, s.wide);
    swiglu_activation(s);
}
__device__ __forceinline__ void swiglu_backward(BlockWorkspace &s) {
    for (int i = threadIdx.x; i < TileTokens * 48; i += BlockThreads) {
        int j = (i / 48) * 96 + i % 48;
        float g = float(s.wide[j]), u = float(s.wide[j + 48]), dy = float(s.work[i]);
        float sig = sigmoid(g);
        bf16 activated = __float2bfloat16_rn(g * sig);
        bf16 da = __float2bfloat16_rn(dy * u);
        s.wide[j] = __float2bfloat16_rn(float(da) * sig * (1.f + g * (1.f - sig)));
        s.wide[j + 48] = __float2bfloat16_rn(dy * float(activated));
    }
    __syncthreads();
}
