#pragma once
#include "types.cuh"

// A 16x8 output tile per warp, BF16 inputs and FP32 accumulation (SM80+).
// Explicit fragment layout lets the epilogue write BF16 directly, without an
// intermediate FP32 shared tile or a block-wide synchronization before casting.
__device__ __forceinline__ unsigned load_pair(const bf16 *p) {
    return *reinterpret_cast<const unsigned *>(p);
}
__device__ __forceinline__ unsigned strided_pair(const bf16 *p, int stride) {
    if (stride == 1)
        return load_pair(p);
    return unsigned(__bfloat16_as_ushort(p[0])) | (unsigned(__bfloat16_as_ushort(p[stride])) << 16);
}
// ldmatrix performs the shared-memory gather/transpose in hardware. Matrix
// strides are in elements; exactly one of each pair of strides equals one.
__device__ __forceinline__ void load_a(unsigned (&a)[4], const bf16 *p, int rows, int cols, int row,
                                       int inner) {
    int lane = threadIdx.x % 32;
    if (__isShared(p)) {
        if (cols == 1) {
            unsigned address = unsigned(
                __cvta_generic_to_shared(p + (row + lane % 16) * rows + inner + (lane / 16) * 8));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(address)
                         : "memory");
        } else {
            unsigned address = unsigned(__cvta_generic_to_shared(
                p + row + ((lane / 8) % 2) * 8 + (inner + lane % 8 + (lane / 16) * 8) * cols));
            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                         : "r"(address)
                         : "memory");
        }
    } else {
        const bf16 *base = p + (row + lane / 4) * rows + (inner + (lane % 4) * 2) * cols;
        a[0] = strided_pair(base, cols);
        a[1] = strided_pair(base + 8 * rows, cols);
        a[2] = strided_pair(base + 8 * cols, cols);
        a[3] = strided_pair(base + 8 * rows + 8 * cols, cols);
    }
}
__device__ __forceinline__ void load_b(unsigned (&b)[2], const bf16 *p, int rows, int cols,
                                       int inner, int col) {
    int lane = threadIdx.x % 32;
    if (__isShared(p)) {
        if (rows == 1) {
            unsigned address = unsigned(__cvta_generic_to_shared(p + inner + ((lane / 8) % 2) * 8 +
                                                                 (col + lane % 8) * cols));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b[0]), "=r"(b[1])
                         : "r"(address)
                         : "memory");
        } else {
            unsigned address =
                unsigned(__cvta_generic_to_shared(p + (inner + lane % 16) * rows + col));
            asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
                         : "=r"(b[0]), "=r"(b[1])
                         : "r"(address)
                         : "memory");
        }
    } else {
        const bf16 *base = p + (inner + (lane % 4) * 2) * rows + (col + lane / 4) * cols;
        b[0] = strided_pair(base, rows);
        b[1] = strided_pair(base + 8 * rows, rows);
    }
}
__device__ __forceinline__ void mma_16x8(float (&c)[4], const unsigned (&a)[4],
                                         const unsigned (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

template <int In, int Out, int Stride = Out>
__device__ __forceinline__ void project(const bf16 *x, const bf16 *weights, bf16 *y,
                                        const bf16 *residual = nullptr) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int row_group = lane / 4, pair = (lane % 4) * 2;
    for (int tile = warp; tile < (TileTokens / 16) * (Out / 8); tile += BlockWarps) {
        int row = tile / (Out / 8) * 16, col = tile % (Out / 8) * 8;
        float c[4] = {};
#pragma unroll
        for (int k = 0; k < In; k += 16) {
            unsigned af[4], bf[2];
            load_a(af, x, In, 1, row, k);
            load_b(bf, weights, 1, In, k, col);
            mma_16x8(c, af, bf);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            int index = (row + row_group + half * 8) * Stride + col + pair;
            auto value = __floats2bfloat162_rn(c[half * 2], c[half * 2 + 1]);
            if (residual)
                value = __hadd2(value, *reinterpret_cast<const __nv_bfloat162 *>(residual + index));
            *reinterpret_cast<__nv_bfloat162 *>(y + index) = value;
        }
    }
    __syncthreads();
}

template <int In, int Out, int Stride = Out>
__device__ __forceinline__ void project(const DeviceState &d, const bf16 *x, int weight, bf16 *y,
                                        const bf16 *residual = nullptr) {
    project<In, Out, Stride>(x, d.weight_high + weight, y, residual);
}
