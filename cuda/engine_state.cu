// Host transfers for model weights, optimizer buffers, and resumed pipeline state.
#include "../runtime.h"
#include "engine_internal.h"
#include <cstring>
#include <vector>

TG_API int tg_set_weights(Engine *e, const float *weights) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        int n = e->d.layout.count;
        e->d.weight_high = e->d.weight_versions;
        e->d.weight_low = e->d.low_versions;
        if (e->d.pipeline)
            check_cuda(cudaMemsetAsync(e->d.pipeline, 0, sizeof(PipelineState), e->stream));
        std::vector<uint16_t> high(n), low(n);
        for (int i = 0; i < n; ++i) {
            uint32_t bits;
            memcpy(&bits, weights + i, 4);
            high[i] = bits >> 16;
            low[i] = uint16_t(bits);
        }
        check_cuda(cudaMemcpyAsync(e->d.weight_high, high.data(), n * 2, cudaMemcpyHostToDevice,
                                   e->stream));
        check_cuda(
            cudaMemcpyAsync(e->d.weight_low, low.data(), n * 2, cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaMemsetAsync(e->d.momentum, 0, n * 4, e->stream));
        check_cuda(cudaMemsetAsync(e->d.variance, 0, n * 4, e->stream));
        e->d.step = 0;
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
TG_API int tg_get_weights(Engine *e, float *weights) {
    try {
        check_cuda(cudaSetDevice(e->d.config.device));
        int n = e->d.layout.count;
        std::vector<uint16_t> high(n), low(n);
        check_cuda(cudaMemcpyAsync(high.data(), e->d.weight_high, n * 2, cudaMemcpyDeviceToHost,
                                   e->stream));
        check_cuda(
            cudaMemcpyAsync(low.data(), e->d.weight_low, n * 2, cudaMemcpyDeviceToHost, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        for (int i = 0; i < n; ++i) {
            uint32_t bits = (uint32_t(high[i]) << 16) | low[i];
            memcpy(weights + i, &bits, 4);
        }
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}

TG_API int tg_get_optimizer_state(Engine *e, float *momentum, float *variance, uint64_t *step) {
    try {
        if (!momentum || !variance || !step)
            throw std::runtime_error("Missing optimizer state output");
        check_cuda(cudaSetDevice(e->d.config.device));
        int count = e->d.layout.count;
        check_cuda(cudaMemcpyAsync(momentum, e->d.momentum, size_t(count) * 4,
                                   cudaMemcpyDeviceToHost, e->stream));
        check_cuda(cudaMemcpyAsync(variance, e->d.variance, size_t(count) * 4,
                                   cudaMemcpyDeviceToHost, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        *step = e->d.step;
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}

TG_API int tg_get_training_losses(Engine *e, float *losses, int count) {
    try {
        if (!losses && count)
            throw std::runtime_error("Missing training loss output");
        if (count < 0 || count > e->last_train_steps)
            throw std::runtime_error("Training loss count exceeds the last launch");
        check_cuda(cudaSetDevice(e->d.config.device));
        check_cuda(cudaMemcpyAsync(losses, e->d.loss_history,
                                   size_t(count) * e->d.config.context * sizeof(float),
                                   cudaMemcpyDeviceToHost, e->stream));
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}

TG_API int tg_set_optimizer_state(Engine *e, const float *momentum, const float *variance,
                                  uint64_t step) {
    try {
        if (!momentum || !variance)
            throw std::runtime_error("Missing optimizer state input");
        check_cuda(cudaSetDevice(e->d.config.device));
        int count = e->d.layout.count;
        check_cuda(cudaMemcpyAsync(e->d.momentum, momentum, size_t(count) * 4,
                                   cudaMemcpyHostToDevice, e->stream));
        check_cuda(cudaMemcpyAsync(e->d.variance, variance, size_t(count) * 4,
                                   cudaMemcpyHostToDevice, e->stream));
        if (e->d.config.inflight > 1) {
            const int versions = e->d.config.inflight + 1;
            for (int version = 1; version < versions; ++version) {
                check_cuda(cudaMemcpyAsync(e->d.weight_versions + size_t(version) * count,
                                           e->d.weight_versions, size_t(count) * 2,
                                           cudaMemcpyDeviceToDevice, e->stream));
                check_cuda(cudaMemcpyAsync(e->d.low_versions + size_t(version) * count,
                                           e->d.low_versions, size_t(count) * 2,
                                           cudaMemcpyDeviceToDevice, e->stream));
            }
            PipelineState pipeline{};
            pipeline.completed = int(step);
            check_cuda(cudaMemcpyAsync(e->d.pipeline, &pipeline, sizeof(pipeline),
                                       cudaMemcpyHostToDevice, e->stream));
            const int current = int(step % uint64_t(versions));
            e->d.weight_high = e->d.weight_versions + size_t(current) * count;
            e->d.weight_low = e->d.low_versions + size_t(current) * count;
        }
        e->d.step = step;
        check_cuda(cudaStreamSynchronize(e->stream));
        return 0;
    } catch (const std::exception &error) {
        tg_last_error = error.what();
        return -1;
    }
}
