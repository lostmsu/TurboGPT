// CUDA resource ownership, launches, and the small C interface used by tests.
#include "../runtime.h"
#include "trainer.cuh"
#if TURBOGPT_PIPELINE
#include "pipeline.cuh"
#endif
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

struct Engine {
    DeviceState d{};
    const void *kernel = nullptr, *pipeline_kernel = nullptr;
    int pipeline_blocks = 0, last_train_steps = 0;
    int schedule_steps = 0;
    int64_t *endpoint_buffer = nullptr;
    uint8_t *input_buffer = nullptr, *target_buffer = nullptr;
    int shared = 0;
    cudaStream_t stream = nullptr;
    std::vector<void *> allocations;
    template <class T> void allocate(T *&ptr, size_t n, bool zero = true) {
        void *raw = nullptr;
        auto e = cudaMalloc(&raw, n * sizeof(T));
        if (e != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(e));
        ptr = static_cast<T *>(raw);
        allocations.push_back(raw);
        if (zero)
            cudaMemsetAsync(raw, 0, n * sizeof(T), stream);
    }
    ~Engine() {
        cudaSetDevice(d.config.device);
        if (stream)
            cudaStreamSynchronize(stream);
        for (void *p : allocations)
            cudaFree(p);
        if (stream)
            cudaStreamDestroy(stream);
    }
};
static thread_local std::string last_error;
static void check(cudaError_t e) {
    if (e != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(e));
}
TG_API const char *tg_error() {
    return last_error.c_str();
}
TG_API int tg_onecycle_point(const OneCycle *schedule, int step, StepRates *rates) {
    try {
        *rates = onecycle_point(*schedule, step);
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_set_onecycle(Engine *e, const OneCycle *schedule) {
    try {
        check(cudaSetDevice(e->d.config.device));
        if (e->d.schedule || e->d.step)
            throw std::runtime_error("Set OneCycle once, before training");
        onecycle_point(*schedule, 0); // Validate before allocating.
        std::vector<StepRates> rates(schedule->total_steps);
        for (int step = 0; step < schedule->total_steps; ++step)
            rates[step] = onecycle_point(*schedule, step);
        e->allocate(e->d.schedule, rates.size(), false);
        check(cudaMemcpyAsync(e->d.schedule, rates.data(), rates.size() * sizeof(StepRates),
                              cudaMemcpyHostToDevice, e->stream));
        check(cudaStreamSynchronize(e->stream));
        e->schedule_steps = schedule->total_steps;
        e->d.cycle_momentum = schedule->cycle_momentum;
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API Engine *tg_create(const Config *c) {
    Engine *e = nullptr;
    try {
        if ((c->context != 4 && c->context != 8) || (c->depth != 4 && c->depth != 8) ||
            c->batch < 4 || c->batch * c->context % 16)
            throw std::runtime_error("Supported: depth 4/8, context 4/8, positive batch with "
                                     "batch*context divisible by 16");
        if (c->inflight < 1 || c->inflight > MaxInFlight)
            throw std::runtime_error("inflight must be 1..256");
#if !TURBOGPT_PIPELINE
        if (c->inflight > 1)
            throw std::runtime_error("Experimental overlap requires build.ps1 -Pipeline");
#endif
        if (c->inflight > 1 && c->blocks)
            throw std::runtime_error("--blocks applies to synchronous training only");
        if (c->ns_steps < 1 || c->ns_steps > 10)
            throw std::runtime_error("ns_steps must be 1..10");
        check(cudaSetDevice(c->device));
        cudaDeviceProp prop{};
        check(cudaGetDeviceProperties(&prop, c->device));
        if (!prop.cooperativeLaunch || prop.major < 8)
            throw std::runtime_error("Requires cooperative launch and BF16 tensor cores (SM80+)");
        e = new Engine;
        e->d.config = *c;
        e->d.layout = parameter_layout(c->depth, c->context);
        check(cudaStreamCreate(&e->stream));
        e->kernel = c->context == 4 ? (void *)persistent<4> : (void *)persistent<8>;
#if TURBOGPT_PIPELINE
        e->pipeline_kernel = c->context == 4 ? (void *)pipelined<4> : (void *)pipelined<8>;
#endif
        e->shared = sizeof(BlockWorkspace);
        check(cudaFuncSetAttribute(e->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   e->shared));
        int active = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, e->kernel, BlockThreads,
                                                            e->shared));
        int limit = active * prop.multiProcessorCount;
        int tiles = (c->batch * c->context + TileTokens - 1) / TileTokens;
        // Balanced work avoids a long final wave with only a few active blocks.
        int balanced = std::min(limit, tiles);
        while (tiles % balanced)
            --balanced;
        e->d.blocks = c->blocks ? c->blocks : balanced;
        if (e->d.blocks < 1 || e->d.blocks > limit)
            throw std::runtime_error("Requested block count exceeds cooperative residency limit");
        e->d.rank = -1;
        e->d.norm_blocks = e->d.blocks;
        int allocation_blocks = e->d.blocks;
        if (c->inflight > 1) {
            check(cudaFuncSetAttribute(e->pipeline_kernel,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize, e->shared));
            check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, e->pipeline_kernel,
                                                                BlockThreads, e->shared));
            limit = active * prop.multiProcessorCount;
            e->d.optimizer_blocks = c->muon ? c->depth * 5 : 16;
            int workers = std::min(tiles, (limit - e->d.optimizer_blocks) / c->inflight);
            if (workers < 1)
                throw std::runtime_error("inflight exceeds cooperative residency; reduce it");
            while (tiles % workers)
                --workers;
            e->d.worker_blocks = workers;
            e->pipeline_blocks = workers * c->inflight + e->d.optimizer_blocks;
            allocation_blocks = std::max(allocation_blocks, e->pipeline_blocks);
            e->allocate(e->d.pipeline, 1);
            e->allocate(e->d.version_trace, MaxLaunchSteps);
        }
        int p = e->d.layout.count;
        e->allocate(e->d.saved, size_t(allocation_blocks) * c->depth, false);
        if (prop.l2CacheSize >= 64 * 1024 * 1024)
            e->allocate(e->d.mlp_cache, size_t(allocation_blocks) * c->depth * TileTokens * 96,
                        false);
        int versions = c->inflight > 1 ? c->inflight + 1 : 1;
        e->allocate(e->d.weight_versions, size_t(p) * versions);
        e->allocate(e->d.low_versions, size_t(p) * versions);
        e->d.weight_high = e->d.weight_versions;
        e->d.weight_low = e->d.low_versions;
        e->allocate(e->d.momentum, p);
        e->allocate(e->d.variance, p);
        e->allocate(e->d.gradient, size_t(p) * c->inflight);
        e->allocate(e->d.partial_gradients, size_t(p) * allocation_blocks);
        e->allocate(e->d.gradient_norms, allocation_blocks);
        e->allocate(e->d.token_losses, size_t(c->batch) * c->context * c->inflight);
        e->allocate(e->endpoint_buffer, c->batch);
        e->allocate(e->input_buffer, c->batch * c->context);
        e->allocate(e->target_buffer, c->batch * c->context);
        check(cudaStreamSynchronize(e->stream));
        return e;
    } catch (const std::exception &ex) {
        last_error = ex.what();
        delete e;
        return nullptr;
    }
}
TG_API void tg_destroy(Engine *e) {
    delete e;
}
TG_API int tg_parameter_count(Engine *e) {
    return e->d.layout.count;
}
TG_API int tg_blocks(Engine *e) {
    return e->d.config.inflight > 1 ? e->pipeline_blocks : e->d.blocks;
}
TG_API int tg_shared_bytes(Engine *e) {
    return e->shared;
}
TG_API int tg_synchronize(Engine *e) {
    try {
        check(cudaSetDevice(e->d.config.device));
        check(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_set_weights(Engine *e, const float *weights) {
    try {
        check(cudaSetDevice(e->d.config.device));
        int n = e->d.layout.count;
        e->d.weight_high = e->d.weight_versions;
        e->d.weight_low = e->d.low_versions;
        if (e->d.pipeline)
            check(cudaMemsetAsync(e->d.pipeline, 0, sizeof(PipelineState), e->stream));
        std::vector<uint16_t> high(n), low(n);
        for (int i = 0; i < n; ++i) {
            uint32_t bits;
            memcpy(&bits, weights + i, 4);
            high[i] = bits >> 16;
            low[i] = uint16_t(bits);
        }
        check(cudaMemcpyAsync(e->d.weight_high, high.data(), n * 2, cudaMemcpyHostToDevice,
                              e->stream));
        check(
            cudaMemcpyAsync(e->d.weight_low, low.data(), n * 2, cudaMemcpyHostToDevice, e->stream));
        check(cudaMemsetAsync(e->d.momentum, 0, n * 4, e->stream));
        check(cudaMemsetAsync(e->d.variance, 0, n * 4, e->stream));
        e->d.step = 0;
        check(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_get_weights(Engine *e, float *weights) {
    try {
        check(cudaSetDevice(e->d.config.device));
        int n = e->d.layout.count;
        std::vector<uint16_t> high(n), low(n);
        check(cudaMemcpyAsync(high.data(), e->d.weight_high, n * 2, cudaMemcpyDeviceToHost,
                              e->stream));
        check(
            cudaMemcpyAsync(low.data(), e->d.weight_low, n * 2, cudaMemcpyDeviceToHost, e->stream));
        check(cudaStreamSynchronize(e->stream));
        for (int i = 0; i < n; ++i) {
            uint32_t bits = (uint32_t(high[i]) << 16) | low[i];
            memcpy(weights + i, &bits, 4);
        }
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_set_data(Engine *e, const uint8_t *bytes, int64_t size) {
    try {
        check(cudaSetDevice(e->d.config.device));
        if (size <= 8)
            throw std::runtime_error("Dataset needs at least 9 bytes");
        if (e->d.data)
            throw std::runtime_error("Dataset already loaded");
        e->allocate(e->d.data, size, false); // copy overwrites every byte
        e->d.size = size;
        check(cudaMemcpyAsync(e->d.data, bytes, size, cudaMemcpyHostToDevice, e->stream));
        check(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
static void launch(Engine *e, DeviceState d, int steps, RunMode mode) {
    check(cudaSetDevice(d.config.device));
    void *args[] = {&d, &steps, &mode};
    check(cudaLaunchCooperativeKernel(e->kernel, dim3(d.blocks), dim3(BlockThreads), args,
                                      e->shared, e->stream));
}
TG_API int tg_train(Engine *e, int steps, float multiplier) {
    try {
        if (!e->d.data || steps < 1 || steps > MaxLaunchSteps)
            throw std::runtime_error("Load data first; each invocation permits 1..4096 steps");
        if (e->d.step + steps > INT_MAX)
            throw std::runtime_error("Training step count exceeds signed 32-bit indexing");
        if (e->d.schedule && e->d.step + steps > uint64_t(e->schedule_steps))
            throw std::runtime_error("Training would exceed the OneCycle schedule");
        e->d.lr_multiplier = multiplier;
        if (e->d.config.inflight > 1) {
            check(cudaSetDevice(e->d.config.device));
            void *args[] = {&e->d, &steps};
            check(cudaLaunchCooperativeKernel(e->pipeline_kernel, dim3(e->pipeline_blocks),
                                              dim3(BlockThreads), args, e->shared, e->stream));
        } else {
            launch(e, e->d, steps, RunMode::Train);
        }
        e->last_train_steps = steps;
        e->d.step += steps;
        if (e->d.config.inflight > 1) {
            size_t offset = (e->d.step % (e->d.config.inflight + 1)) * e->d.layout.count;
            e->d.weight_high = e->d.weight_versions + offset;
            e->d.weight_low = e->d.low_versions + offset;
        }
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_training_versions(Engine *e, int *out, int count) {
    try {
        if (count < 0 || count > e->last_train_steps)
            throw std::runtime_error("Version trace exceeds last training launch");
        if (e->d.config.inflight > 1) {
            check(cudaMemcpyAsync(out, e->d.version_trace, count * sizeof(int),
                                  cudaMemcpyDeviceToHost, e->stream));
            check(cudaStreamSynchronize(e->stream));
        } else {
            for (int i = 0; i < count; ++i)
                out[i] = int(e->d.step) - e->last_train_steps + i;
        }
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_pipeline_stats(Engine *e, double *mean, int *maximum) {
    try {
        *mean = 0;
        *maximum = 0;
        if (e->d.pipeline && e->d.step) {
            PipelineState state{};
            check(cudaMemcpyAsync(&state, e->d.pipeline, sizeof(state), cudaMemcpyDeviceToHost,
                                  e->stream));
            check(cudaStreamSynchronize(e->stream));
            *mean = double(state.total_staleness) / e->d.step;
            *maximum = state.max_staleness;
        }
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
static void read_losses(Engine *e, float *out) {
    std::vector<float> losses(e->d.config.batch * e->d.config.context);
    check(cudaMemcpyAsync(losses.data(), e->d.token_losses, losses.size() * 4,
                          cudaMemcpyDeviceToHost, e->stream));
    check(cudaStreamSynchronize(e->stream));
    for (int p = 0; p < e->d.config.context; ++p) {
        double sum = 0;
        for (int b = 0; b < e->d.config.batch; ++b)
            sum += losses[b * e->d.config.context + p];
        out[p] = float(sum / e->d.config.batch);
    }
}
TG_API int tg_evaluate(Engine *e, const int64_t *endpoints, float *loss) {
    try {
        check(cudaSetDevice(e->d.config.device));
        if (!e->d.data)
            throw std::runtime_error("Load data before evaluation");
        for (int i = 0; i < e->d.config.batch; ++i)
            if (endpoints[i] < e->d.config.context || endpoints[i] >= e->d.size)
                throw std::runtime_error("Evaluation target endpoint outside the byte stream");
        DeviceState d = e->d;
        d.endpoints = e->endpoint_buffer;
        check(cudaMemcpyAsync((void *)d.endpoints, endpoints, d.config.batch * 8,
                              cudaMemcpyHostToDevice, e->stream));
        launch(e, d, 1, RunMode::Evaluate);
        read_losses(e, loss);
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_batch(Engine *e, const uint8_t *x, const uint8_t *y, float *logits, float *gradient,
                    float *loss) {
    try {
        check(cudaSetDevice(e->d.config.device));
        DeviceState d = e->d;
        int tokens = d.config.batch * d.config.context;
        d.input = e->input_buffer;
        d.target = e->target_buffer;
        check(cudaMemcpyAsync((void *)d.input, x, tokens, cudaMemcpyHostToDevice, e->stream));
        check(cudaMemcpyAsync((void *)d.target, y, tokens, cudaMemcpyHostToDevice, e->stream));
        check(cudaMalloc(&d.logits, size_t(tokens) * 256 * 4));
        try {
            launch(e, d, 1, gradient ? RunMode::Gradients : RunMode::Evaluate);
            check(cudaMemcpyAsync(logits, d.logits, size_t(tokens) * 256 * 4,
                                  cudaMemcpyDeviceToHost, e->stream));
            if (gradient)
                check(cudaMemcpyAsync(gradient, d.gradient, d.layout.count * 4,
                                      cudaMemcpyDeviceToHost, e->stream));
            read_losses(e, loss);
        } catch (...) {
            cudaFree(d.logits);
            throw;
        }
        check(cudaFree(d.logits));
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
TG_API int tg_optimizer_test(Engine *e, const float *gradient, int steps) {
    try {
        check(cudaSetDevice(e->d.config.device));
        if (e->d.config.inflight > 1)
            throw std::runtime_error("Optimizer oracle requires synchronous configuration");
        if (steps < 1 || steps > 64)
            throw std::runtime_error("steps must be 1..64");
        if (e->d.schedule && e->d.step + steps > uint64_t(e->schedule_steps))
            throw std::runtime_error("Optimizer test would exceed the OneCycle schedule");
        check(cudaMemcpyAsync(e->d.gradient, gradient, e->d.layout.count * 4,
                              cudaMemcpyHostToDevice, e->stream));
        e->d.lr_multiplier = 1;
        launch(e, e->d, steps, RunMode::OptimizerOnly);
        e->d.step += steps;
        check(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        last_error = error.what();
        return -1;
    }
}
