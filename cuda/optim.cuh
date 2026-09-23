#pragma once
#include "ops.cuh"

// Mantissa-preserving AdamW for embeddings/head/norms; Muon for hidden matrices.
__device__ __forceinline__ bool matrix_parameter(const DeviceState &d, int i, bool &hidden) {
    hidden = false;
    for (int l = 0; l < d.config.depth; ++l) {
        const LayerParameters &p = d.layout.layers[l];
        if ((i >= p.qkv && i < p.mlp_norm) || (i >= p.gate && i < p.down + 16 * 48)) {
            hidden = true;
            return true;
        }
    }
    return i >= d.layout.head;
}
__device__ __forceinline__ void muon_matrix(const DeviceState &d, BlockWorkspace &s, int weight,
                                            int rows, int cols, float clip, float lr,
                                            float momentum) {
    int n = rows > cols ? rows : cols, size = rows * cols;
    float squares = 0;
    for (int i = threadIdx.x; i < size; i += BlockThreads) {
        float g = d.gradient[weight + i] * clip;
        float m = d.momentum[weight + i] + (1 - momentum) * (g - d.momentum[weight + i]);
        d.momentum[weight + i] = m;
        float u = g + momentum * (m - g);
        bf16 b = __float2bfloat16_rn(u);
        int index = rows > cols ? (i % cols) * rows + i / cols : i;
        s.wide[index] = b;
        squares += float(b) * float(b);
    }
    squares = warp_sum(squares);
    if (threadIdx.x % 32 == 0)
        s.result[threadIdx.x / 32] = squares;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int j = 0; j < BlockWarps; ++j)
            sum += s.result[j];
        s.result[0] =
            fmaxf(float(__float2bfloat16_rn(sqrtf(sum))), float(__float2bfloat16_rn(1e-7f)));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < size; i += BlockThreads)
        s.wide[i] = __float2bfloat16_rn(float(s.wide[i]) / s.result[0]);
    __syncthreads();
    for (int step = 0; step < d.config.ns_steps; ++step) {
        matmul(s, s.wide, n, 1, s.wide, 1, n, 16, 16, n);
        cast_result(s, s.branch, 256); // A = X X^T
        matmul(s, s.branch, 16, 1, s.branch, 16, 1, 16, 16, 16);
        for (int i = threadIdx.x; i < 256; i += BlockThreads)
            s.work[i] = __float2bfloat16_rn(-4.775f * float(s.branch[i]) + 2.0315f * s.result[i]);
        __syncthreads();
        matmul(s, s.work, 16, 1, s.wide, n, 1, 16, n, 16);
        for (int i = threadIdx.x; i < size; i += BlockThreads)
            s.wide[i] = __float2bfloat16_rn(3.4445f * float(s.wide[i]) + s.result[i]);
        __syncthreads();
    }
    float adjusted = lr * sqrtf(fmaxf(1.f, float(rows) / cols));
    for (int i = threadIdx.x; i < size; i += BlockThreads) {
        int index = rows > cols ? (i % cols) * rows + i / cols : i;
        float w = precise_weight(d, weight + i) * (1 - lr * d.config.weight_decay);
        write_weight(d, weight + i, w - adjusted * float(s.wide[index]));
    }
    __syncthreads();
}
__device__ __forceinline__ void optimize(const DeviceState &d, BlockWorkspace &s, uint64_t step) {
    float local_norm = 0;
    for (int b = threadIdx.x; b < d.norm_blocks; b += BlockThreads)
        local_norm += d.gradient_norms[b];
    local_norm = warp_sum(local_norm);
    if (threadIdx.x % 32 == 0)
        s.result[threadIdx.x / 32] = local_norm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float norm = 0;
        for (int warp = 0; warp < BlockWarps; ++warp)
            norm += s.result[warp];
        s.result[0] = d.config.clip > 0 ? fminf(1.f, d.config.clip / (sqrtf(norm) + 1e-6f)) : 1.f;
    }
    __syncthreads();
    float multiplier = d.lr_multiplier, beta1 = d.config.beta1, momentum = d.config.momentum;
    if (d.schedule) {
        // Every optimizer update, including updates inside a persistent launch,
        // reads its own schedule point. Host chunk size does not quantize LR.
        StepRates rates = d.schedule[step];
        multiplier *= rates.multiplier;
        if (d.cycle_momentum)
            beta1 = momentum = rates.momentum;
    }
    float clip = s.result[0], lr = d.config.learning_rate * multiplier;
    // Match torch.optim.AdamW's current-beta bias correction under cycling.
    float bc1 = 1 - powf(beta1, float(step + 1)), bc2 = 1 - powf(d.config.beta2, float(step + 1));
    for (int i = block_rank(d) * BlockThreads + threadIdx.x; i < d.layout.count;
         i += d.blocks * BlockThreads) {
        bool hidden;
        bool decay = matrix_parameter(d, i, hidden);
        if (d.config.muon && hidden)
            continue;
        float g = d.gradient[i] * clip;
        float m = d.momentum[i] + (1 - beta1) * (g - d.momentum[i]);
        float v = d.config.beta2 * d.variance[i] + (1 - d.config.beta2) * g * g;
        d.momentum[i] = m;
        d.variance[i] = v;
        float w = precise_weight(d, i) * (1 - lr * (decay ? d.config.weight_decay : 0.f));
        write_weight(d, i, w - (lr / bc1) * m / (sqrtf(v) / sqrtf(bc2) + d.config.epsilon));
    }
    __syncthreads();
    if (d.config.muon)
        for (int index = block_rank(d); index < d.config.depth * 5; index += d.blocks) {
            const LayerParameters &p = d.layout.layers[index / 5];
            int type = index % 5;
            int weight = type == 0   ? p.qkv
                         : type == 1 ? p.attention
                         : type == 2 ? p.gate
                         : type == 3 ? p.up
                                     : p.down;
            int rows = type == 0 || type == 2 || type == 3 ? 48 : 16;
            int cols = type == 4 ? 48 : 16;
            muon_matrix(d, s, weight, rows, cols, clip, d.config.muon_lr * multiplier, momentum);
        }
}
