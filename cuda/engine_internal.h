#pragma once
#include "trainer.cuh"
#include <string>
#include <vector>

struct Engine {
    DeviceState d{};
    const void *kernel = nullptr, *pipeline_kernel = nullptr;
    int pipeline_blocks = 0, last_train_steps = 0;
    int schedule_steps = 0;
    uint8_t *input_buffer = nullptr, *target_buffer = nullptr;
    float *logits_buffer = nullptr;
    int shared = 0;
    cudaStream_t stream = nullptr;
    std::vector<void *> allocations;
    size_t allocated_bytes = 0;
    template <class T> void allocate(T *&ptr, size_t n, bool zero = true) {
        void *raw = nullptr;
        auto error = cudaMalloc(&raw, n * sizeof(T));
        if (error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(error));
        ptr = static_cast<T *>(raw);
        allocations.push_back(raw);
        allocated_bytes += n * sizeof(T);
        if (zero)
            cudaMemsetAsync(raw, 0, n * sizeof(T), stream);
    }
    ~Engine() {
        cudaSetDevice(d.config.device);
        if (stream)
            cudaStreamSynchronize(stream);
        for (void *allocation : allocations)
            cudaFree(allocation);
        if (stream)
            cudaStreamDestroy(stream);
    }
};

extern thread_local std::string tg_last_error;
inline void check_cuda(cudaError_t error) {
    if (error != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(error));
}
