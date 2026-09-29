// CUDA resource ownership, launches, and the small C interface used by tests.
#include "../runtime.h"
#include "engine_internal.h"
#include "trainer.cuh"
#if TURBOGPT_PIPELINE
#include "pipeline.cuh"
#endif
#include <algorithm>
#include <string>
#include <vector>

thread_local std::string tg_last_error;
static int tile_count(const Config &config) {
    int64_t tokens = int64_t(config.batch) * config.context;
    return int((tokens + TileTokens - 1) / TileTokens);
}
static void require_balanced_tiles(const Config &config, int tiles, int teams,
                                   const char *team_name) {
    if (tiles % teams == 0)
        return;
    int64_t balanced_tiles = (int64_t(tiles) + teams - 1) / teams * teams;
    int suggested_batch = int(balanced_tiles * TileTokens / config.context);
    throw std::runtime_error("Batch " + std::to_string(config.batch) + " x ctx" +
                             std::to_string(config.context) + " creates " + std::to_string(tiles) +
                             " token tiles, which cannot evenly fill " + std::to_string(teams) +
                             " resident " + team_name +
                             ". Edit experiment.h; the nearest balanced batch is " +
                             std::to_string(suggested_batch) + " for optimal occupancy.");
}
TG_API const char *tg_error() {
    return tg_last_error.c_str();
}
TG_API int tg_onecycle_point(const OneCycle *schedule, int step, StepRates *rates) {
    try {
        *rates = onecycle_point(*schedule, step);
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_set_onecycle(Engine *e, const OneCycle *schedule) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        if (e->d.schedule || e->d.step)
            throw std::runtime_error("Set OneCycle once, before training");
        onecycle_point(*schedule, 0); // Validate before allocating.
        std::vector<StepRates> rates(schedule->total_steps);
        for (int step = 0; step < schedule->total_steps; ++step)
            rates[step] = onecycle_point(*schedule, step);
        e->allocate(e->d.schedule, rates.size(), false);
        check_cuda(cudaMemcpyAsync(e->d.schedule, rates.data(), rates.size() * sizeof(StepRates),
                                   cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        e->schedule_steps = schedule->total_steps;
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API Engine *tg_create(const Config *c) {
    Engine *e = nullptr;
    try {
        if ((c->context != 4 && c->context != 8) || (c->depth != 4 && c->depth != 8) ||
            c->batch < 4 || c->batch * c->context % 16)
            throw std::runtime_error("Supported: depth 4/8, context 4/8, batch at least 4 with "
                                     "batch*context divisible by 16");
        if (c->inflight < 1 || c->inflight > MaxInFlight)
            throw std::runtime_error("inflight must be 1..256");
#if !TURBOGPT_PIPELINE
        if (c->inflight > 1)
            throw std::runtime_error("Experimental overlap requires build.ps1 -Pipeline");
#endif
        if (c->inflight > 1 && c->blocks)
            throw std::runtime_error("model.blocks applies to synchronous training only");
        if (c->ns_steps < 1 || c->ns_steps > 10)
            throw std::runtime_error("ns_steps must be 1..10");
        check_cuda(cudaSetDevice(c->device));
        cudaDeviceProp prop{};
        check_cuda(cudaGetDeviceProperties(&prop, c->device));
        if (!prop.cooperativeLaunch || prop.major < 8)
            throw std::runtime_error("Requires cooperative launch and BF16 tensor cores (SM80+)");
        e = new Engine;
        e->d.config = *c;
        e->d.layout = parameter_layout(c->depth, c->context);
        check_cuda(cudaStreamCreate(&e->stream));
        e->kernel = c->context == 4 ? (void *)persistent<4> : (void *)persistent<8>;
#if TURBOGPT_PIPELINE
        e->pipeline_kernel = c->context == 4 ? (void *)pipelined<4> : (void *)pipelined<8>;
#endif
        e->shared = sizeof(BlockWorkspace);
        check_cuda(cudaFuncSetAttribute(e->kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        e->shared));
        int active = 0;
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, e->kernel, BlockThreads,
                                                                 e->shared));
        int limit = active * prop.multiProcessorCount;
        int tiles = tile_count(*c);
        if (limit < 1)
            throw std::runtime_error("Kernel has no cooperative residency on this device");
        int balanced = std::min(limit, tiles);
        if (c->inflight == 1)
            require_balanced_tiles(*c, tiles, balanced, "blocks");
        e->d.blocks = c->blocks ? c->blocks : balanced;
        if (e->d.blocks < 1 || e->d.blocks > limit)
            throw std::runtime_error("Requested block count exceeds cooperative residency limit");
        e->d.rank = -1;
        e->d.norm_blocks = e->d.blocks;
        int allocation_blocks = e->d.blocks;
        if (c->inflight > 1) {
            check_cuda(cudaFuncSetAttribute(
                e->pipeline_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, e->shared));
            check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, e->pipeline_kernel,
                                                                     BlockThreads, e->shared));
            limit = active * prop.multiProcessorCount;
            e->d.optimizer_blocks = c->depth * 5;
            int resident_workers = (limit - e->d.optimizer_blocks) / c->inflight;
            int workers = std::min(tiles, resident_workers);
            if (workers < 1)
                throw std::runtime_error("inflight exceeds cooperative residency; reduce it");
            require_balanced_tiles(*c, tiles, workers, "worker teams");
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
        e->allocate(e->d.loss_history, size_t(MaxLaunchSteps) * c->context);
        e->allocate(e->input_buffer, c->batch * c->context);
        e->allocate(e->target_buffer, c->batch * c->context);
        check_cuda(cudaStreamSynchronize(e->stream));
        return e;
    } catch (const std::exception &ex) {
        tg_last_error = ex.what();
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
        check_cuda(cudaSetDevice(e->d.config.device));
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_memory(Engine *e, double *used_gb) {
    try {
        if (!used_gb)
            throw std::runtime_error("Missing memory output");
        check_cuda(cudaSetDevice(e->d.config.device));
        *used_gb = double(e->allocated_bytes) / (1024.0 * 1024.0 * 1024.0);
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_set_data(Engine *e, const uint8_t *bytes, int64_t size) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        if (size <= 8)
            throw std::runtime_error("Dataset needs at least 9 bytes");
        if (e->d.data)
            throw std::runtime_error("Dataset already loaded");
        e->allocate(e->d.data, size, false); // copy overwrites every byte
        e->d.size = size;
        check_cuda(cudaMemcpyAsync(e->d.data, bytes, size, cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
static void launch(Engine *e, DeviceState d, int steps, RunMode mode) {
    check_cuda(cudaSetDevice(d.config.device));
    void *args[] = {&d, &steps, &mode};
    check_cuda(cudaLaunchCooperativeKernel(e->kernel, dim3(d.blocks), dim3(BlockThreads), args,
                                           e->shared, e->stream));
}
TG_API int tg_train(Engine *e, int steps) {
    try {
        if (!e->d.data || steps < 1 || steps > MaxLaunchSteps)
            throw std::runtime_error("Load data first; each invocation permits 1.." +
                                     std::to_string(MaxLaunchSteps) + " steps");
        if (e->d.step + steps > INT_MAX)
            throw std::runtime_error("Training step count exceeds signed 32-bit indexing");
        if (e->d.schedule && e->d.step + steps > uint64_t(e->schedule_steps))
            throw std::runtime_error("Training would exceed the OneCycle schedule");
        if (e->d.config.inflight > 1) {
            check_cuda(cudaSetDevice(e->d.config.device));
            void *args[] = {&e->d, &steps};
            check_cuda(cudaLaunchCooperativeKernel(e->pipeline_kernel, dim3(e->pipeline_blocks),
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
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_training_versions(Engine *e, int *out, int count) {
    try {
        if (count < 0 || count > e->last_train_steps)
            throw std::runtime_error("Version trace exceeds last training launch");
        if (e->d.config.inflight > 1) {
            check_cuda(cudaMemcpyAsync(out, e->d.version_trace, count * sizeof(int),
                                       cudaMemcpyDeviceToHost, e->stream));
            check_cuda(cudaStreamSynchronize(e->stream));
        } else {
            for (int i = 0; i < count; ++i)
                out[i] = int(e->d.step) - e->last_train_steps + i;
        }
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_pipeline_stats(Engine *e, double *mean, int *maximum) {
    try {
        *mean = 0;
        *maximum = 0;
        if (e->d.pipeline && e->d.step) {
            PipelineState state{};
            check_cuda(cudaMemcpyAsync(&state, e->d.pipeline, sizeof(state), cudaMemcpyDeviceToHost,
                                       e->stream));
            check_cuda(cudaStreamSynchronize(e->stream));
            *mean = double(state.total_staleness) / e->d.step;
            *maximum = state.max_staleness;
        }
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
static void read_losses(Engine *e, float *out) {
    std::vector<float> losses(e->d.config.batch * e->d.config.context);
    check_cuda(cudaMemcpyAsync(losses.data(), e->d.token_losses, losses.size() * 4,
                               cudaMemcpyDeviceToHost, e->stream));
    check_cuda(cudaStreamSynchronize(e->stream));
    for (int p = 0; p < e->d.config.context; ++p) {
        double sum = 0;
        for (int b = 0; b < e->d.config.batch; ++b)
            sum += losses[b * e->d.config.context + p];
        out[p] = float(sum / e->d.config.batch);
    }
}
TG_API int tg_batch(Engine *e, const uint8_t *x, const uint8_t *y, float *logits, float *gradient,
                    float *loss) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        DeviceState d = e->d;
        int tokens = d.config.batch * d.config.context;
        d.input = e->input_buffer;
        d.target = e->target_buffer;
        check_cuda(cudaMemcpyAsync((void *)d.input, x, tokens, cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaMemcpyAsync((void *)d.target, y, tokens, cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaMalloc(&d.logits, size_t(tokens) * 256 * 4));
        try {
            launch(e, d, 1, gradient ? RunMode::Gradients : RunMode::Evaluate);
            check_cuda(cudaMemcpyAsync(logits, d.logits, size_t(tokens) * 256 * 4,
                                       cudaMemcpyDeviceToHost, e->stream));
            if (gradient)
                check_cuda(cudaMemcpyAsync(gradient, d.gradient, d.layout.count * 4,
                                           cudaMemcpyDeviceToHost, e->stream));
            read_losses(e, loss);
        } catch (...) {
            cudaFree(d.logits);
            throw;
        }
        check_cuda(cudaFree(d.logits));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}

TG_API int tg_predict(Engine *e, const uint8_t *context, float *logits) {
    try {
        if (!context || !logits)
            throw std::runtime_error("Missing prediction input or output");
        check_cuda(cudaSetDevice(e->d.config.device));
        if (!e->logits_buffer)
            e->allocate(e->logits_buffer, size_t(e->d.config.batch) * e->d.config.context * 256);
        DeviceState d = e->d;
        d.input = e->input_buffer;
        d.target = e->target_buffer;
        d.logits = e->logits_buffer;
        std::vector<uint8_t> input(size_t(d.config.batch) * d.config.context);
        std::vector<uint8_t> target(input.size());
        for (int batch = 0; batch < d.config.batch; ++batch)
            std::copy(context, context + d.config.context,
                      input.begin() + size_t(batch) * d.config.context);
        check_cuda(cudaMemcpyAsync((void *)d.input, input.data(), input.size(),
                                   cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaMemcpyAsync((void *)d.target, target.data(), target.size(),
                                   cudaMemcpyHostToDevice, e->stream));
        launch(e, d, 1, RunMode::Evaluate);
        check_cuda(cudaMemcpyAsync(logits, e->logits_buffer + size_t(d.config.context - 1) * 256,
                                   256 * sizeof(float), cudaMemcpyDeviceToHost, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_optimizer_test(Engine *e, const float *gradient, int steps) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        if (e->d.config.inflight > 1)
            throw std::runtime_error("Optimizer oracle requires synchronous configuration");
        if (steps < 1 || steps > 64)
            throw std::runtime_error("steps must be 1..64");
        if (e->d.schedule && e->d.step + steps > uint64_t(e->schedule_steps))
            throw std::runtime_error("Optimizer test would exceed the OneCycle schedule");
        check_cuda(cudaMemcpyAsync(e->d.gradient, gradient, e->d.layout.count * 4,
                                   cudaMemcpyHostToDevice, e->stream));
        launch(e, e->d, steps, RunMode::OptimizerOnly);
        e->d.step += steps;
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
