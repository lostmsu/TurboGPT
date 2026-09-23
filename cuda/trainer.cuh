#pragma once
#include "dataset.cuh"
#include "model.cuh"
#include "optim.cuh"

enum class RunMode { Evaluate, Gradients, Train, OptimizerOnly };

// Each team member owns disjoint outputs of the gradient reduction.
__device__ __forceinline__ void reduce_gradients(const DeviceState &d, BlockWorkspace &s,
                                                 RunMode mode) {
    float norm = 0;
    // Spread each output group across all resident blocks, with multiple
    // warps reducing independent ranges of partials. A single thread no
    // longer waits for hundreds of dependent loads while most blocks idle.
    constexpr int Outputs = 64, Groups = BlockThreads / Outputs;
    int output = threadIdx.x % Outputs, group = threadIdx.x / Outputs;
    for (int base = block_rank(d) * Outputs; base < d.layout.count; base += d.blocks * Outputs) {
        int i = base + output;
        float sum = 0;
        if (i < d.layout.count && mode == RunMode::OptimizerOnly)
            sum = group == 0 ? d.gradient[i] : 0;
        else if (i < d.layout.count) {
            for (int b = group; b < d.blocks; b += Groups)
                sum += d.partial_gradients[b * d.layout.count + i];
        }
        s.result[threadIdx.x] = sum;
        __syncthreads();
        if (group == 0 && i < d.layout.count) {
            sum = 0;
            for (int g = 0; g < Groups; ++g)
                sum += s.result[g * Outputs + output];
            d.gradient[i] = sum;
            norm += sum * sum;
        }
        __syncthreads();
    }
    norm = warp_sum(norm);
    if (threadIdx.x % 32 == 0)
        s.result[threadIdx.x / 32] = norm;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0;
        for (int j = 0; j < BlockWarps; ++j)
            sum += s.result[j];
        d.gradient_norms[block_rank(d)] = sum;
    }
}

template <int Context>
__device__ __forceinline__ void batch_gradient(const DeviceState &d, BlockWorkspace &s,
                                               uint64_t step, RunMode mode) {
    auto *saved = d.saved + blockIdx.x * d.config.depth;
    float *partial = d.partial_gradients + block_rank(d) * d.layout.count;
    int tiles = (d.config.batch * Context + TileTokens - 1) / TileTokens;
    if (mode != RunMode::Evaluate) {
        int clear = block_rank(d) >= tiles ? d.layout.count : d.layout.position + Context * 16;
        for (int i = threadIdx.x; i < clear; i += BlockThreads)
            partial[i] = 0;
        if (threadIdx.x < 16 && block_rank(d) < tiles) {
            partial[d.layout.final_norm + threadIdx.x] = 0;
            for (int l = 0; l < d.config.depth; ++l) {
                partial[d.layout.layers[l].attention_norm + threadIdx.x] = 0;
                partial[d.layout.layers[l].mlp_norm + threadIdx.x] = 0;
            }
        }
    }
    __syncthreads();
    for (int tile = block_rank(d); tile < tiles; tile += d.blocks) {
        sample<Context>(d, s, tile, step);
        forward<Context>(d, s, saved, tile);
        if (mode != RunMode::Evaluate)
            backward<Context>(d, s, saved, partial, tile == block_rank(d));
    }
}

// One cooperative launch owns complete synchronous optimizer steps.
template <int Context>
__global__ __launch_bounds__(BlockThreads, MinResidentBlocks) void persistent(
    const __grid_constant__ DeviceState d, int steps, RunMode mode) {
    extern __shared__ __align__(32) unsigned char memory[];
    auto &s = *reinterpret_cast<BlockWorkspace *>(memory);
    auto grid = cg::this_grid();
    for (int iteration = 0; iteration < steps; ++iteration) {
        if (mode != RunMode::OptimizerOnly) {
            batch_gradient<Context>(d, s, d.step + iteration, mode);
            if (mode == RunMode::Evaluate)
                return;
            grid.sync();
        }
        reduce_gradients(d, s, mode);
        grid.sync();
        if (mode == RunMode::Gradients)
            return;
        optimize(d, s, d.step + iteration);
        grid.sync();
    }
}
