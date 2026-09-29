#pragma once
#include "trainer.cuh"
#include <cuda/atomic>

// All blocks are resident under a cooperative launch. Worker teams synchronize
// independently; they never wait at a grid barrier with the optimizer team.
__device__ __forceinline__ int load_acquire(int &value) {
    return cuda::atomic_ref<int, cuda::thread_scope_device>(value).load(cuda::memory_order_acquire);
}
__device__ __forceinline__ void store_release(int &value, int next) {
    cuda::atomic_ref<int, cuda::thread_scope_device>(value).store(next, cuda::memory_order_release);
}
__device__ __forceinline__ void team_sync(TeamBarrier &barrier, int blocks) {
    // Fence every writer before its representative announces arrival.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0 && blocks > 1) {
        int epoch = load_acquire(barrier.epoch);
        auto arrivals = cuda::atomic_ref<int, cuda::thread_scope_device>(barrier.arrivals);
        if (arrivals.fetch_add(1, cuda::memory_order_acq_rel) == blocks - 1) {
            arrivals.store(0, cuda::memory_order_relaxed);
            store_release(barrier.epoch, epoch + 1);
        } else {
            while (load_acquire(barrier.epoch) == epoch)
                __nanosleep(64);
        }
    }
    __syncthreads();
}
__device__ __forceinline__ void wait_at_least(int &value, int target) {
    if (threadIdx.x == 0)
        while (load_acquire(value) < target)
            __nanosleep(64);
    __syncthreads();
}

// A batch uses one immutable weight version for its complete forward/backward.
// At most K batches are outstanding. K+1 versions therefore suffice: a version
// cannot be overwritten until every batch that might still read it has finished.
template <int Context>
__global__ __launch_bounds__(BlockThreads, MinResidentBlocks) void pipelined(
    const __grid_constant__ DeviceState initial, int steps) {
    extern __shared__ __align__(32) unsigned char memory[];
    auto &s = *reinterpret_cast<BlockWorkspace *>(memory);
    // One descriptor per block avoids a large private copy in every thread.
    __shared__ DeviceState d;
    if (threadIdx.x == 0)
        d = initial;
    __syncthreads();
    PipelineState &queue = *d.pipeline;
    int slots = d.config.inflight, parameters = d.layout.count;
    int workers = slots * d.worker_blocks;
    if (blockIdx.x < workers) {
        int slot = blockIdx.x / d.worker_blocks;
        if (threadIdx.x == 0) {
            d.rank = blockIdx.x % d.worker_blocks;
            d.blocks = d.worker_blocks;
            d.norm_blocks = d.blocks;
            d.partial_gradients += size_t(slot) * d.blocks * parameters;
            d.gradient += slot * parameters;
            d.gradient_norms += slot * d.blocks;
            d.token_losses += slot * d.config.batch * Context;
        }
        __syncthreads();
        TeamBarrier &barrier = queue.teams[slot];
        for (int iteration = slot; iteration < steps; iteration += slots) {
            int step = int(d.step) + iteration;
            if (d.rank == 0 && threadIdx.x == 0) {
                int version = load_acquire(queue.completed);
                queue.version[slot] = version;
                d.version_trace[iteration] = version;
                atomicAdd(&queue.total_staleness, static_cast<unsigned long long>(step - version));
                atomicMax(&queue.max_staleness, step - version);
            }
            team_sync(barrier, d.blocks);
            int version = queue.version[slot] % (slots + 1);
            if (threadIdx.x == 0) {
                d.weight_high = d.weight_versions + size_t(version) * parameters;
                d.weight_low = d.low_versions + size_t(version) * parameters;
            }
            __syncthreads();
            batch_gradient<Context>(d, s, step, RunMode::Train);
            team_sync(barrier, d.blocks);
            record_loss_history<Context>(d, s, iteration);
            reduce_gradients(d, s, RunMode::Train);
            team_sync(barrier, d.blocks);
            if (d.rank == 0 && threadIdx.x == 0)
                store_release(queue.ready[slot], step + 1);
            // The optimizer must finish consuming this slot before reuse.
            wait_at_least(queue.completed, step + 1);
        }
    } else {
        if (threadIdx.x == 0) {
            d.rank = blockIdx.x - workers;
            d.blocks = d.optimizer_blocks;
            d.norm_blocks = d.worker_blocks;
        }
        __syncthreads();
        float *gradients = d.gradient, *norms = d.gradient_norms;
        TeamBarrier &barrier = queue.teams[slots];
        for (int iteration = 0; iteration < steps; ++iteration) {
            int step = int(d.step) + iteration, slot = iteration % slots;
            wait_at_least(queue.ready[slot], step + 1);
            int current = step % (slots + 1), next = (step + 1) % (slots + 1);
            const bf16 *high = d.weight_versions + size_t(current) * parameters;
            const uint16_t *low = d.low_versions + size_t(current) * parameters;
            if (threadIdx.x == 0) {
                d.weight_high = d.weight_versions + size_t(next) * parameters;
                d.weight_low = d.low_versions + size_t(next) * parameters;
                d.gradient = gradients + slot * parameters;
                d.gradient_norms = norms + slot * d.worker_blocks;
            }
            __syncthreads();
            for (int i = d.rank * BlockThreads + threadIdx.x; i < parameters;
                 i += d.blocks * BlockThreads) {
                d.weight_high[i] = high[i];
                d.weight_low[i] = low[i];
            }
            team_sync(barrier, d.blocks);
            optimize(d, s, step);
            team_sync(barrier, d.blocks);
            if (d.rank == 0 && threadIdx.x == 0)
                store_release(queue.completed, step + 1);
        }
    }
}
