#pragma once
#include "../model.h"
#include "../optim.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <math_constants.h>

namespace cg = cooperative_groups;
constexpr int MaxInFlight = 256, MaxLaunchSteps = 4096;
struct TeamBarrier {
    int arrivals, epoch;
};
struct PipelineState {
    int completed;
    unsigned long long total_staleness;
    int max_staleness;
    int ready[MaxInFlight], version[MaxInFlight];
    TeamBarrier teams[MaxInFlight + 1];
};
using bf16 = __nv_bfloat16;
#ifndef TURBOGPT_THREADS
#define TURBOGPT_THREADS 512
#endif
#ifndef TURBOGPT_TILE
#define TURBOGPT_TILE 32
#endif
constexpr int TileTokens = TURBOGPT_TILE, BlockThreads = TURBOGPT_THREADS,
              BlockWarps = BlockThreads / 32;
// Blackwell benefits from two resident CTAs. Ampere's smaller shared memory
// permits only one of these workspaces; do not constrain its registers to two.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1200
constexpr int MinResidentBlocks = 2;
#else
constexpr int MinResidentBlocks = 1;
#endif
constexpr int HeadStride = 264; // Eight BF16 padding elements break shared-bank conflicts.
// Saved activations for one token tile. The sequence boundary is
// determined by context, not by the matrix-multiply tile.
struct LayerActivations {
    bf16 x[TileTokens * 16], qkv[TileTokens * 48], middle[TileTokens * 16];
    int selected[TileTokens * 4];
    float gates[TileTokens * 4], derivatives[TileTokens * 4];
};
struct alignas(32) BlockWorkspace {
    bf16 hidden[TileTokens * 16], normalized[TileTokens * 16], dhidden[TileTokens * 16],
        branch[TileTokens * 16];
    bf16 wide[TileTokens * HeadStride], work[TileTokens * 64], dqkv[TileTokens * 48];
    float result[TileTokens * 96];
    uint8_t x[TileTokens], y[TileTokens];
};
static_assert(sizeof(LayerActivations) == TileTokens * 208);

struct DeviceState {
    Config config;
    ParameterLayout layout;
    bf16 *weight_high;
    uint16_t *weight_low;
    float *momentum, *variance, *partial_gradients, *gradient, *gradient_norms, *token_losses;
    float *loss_history;
    uint8_t *data;
    int64_t size;
    const uint8_t *input, *target;
    float *logits;
    LayerActivations *saved;
    bf16 *mlp_cache;
    StepRates *schedule;
    int blocks;
    int rank, norm_blocks, worker_blocks, optimizer_blocks;
    PipelineState *pipeline;
    bf16 *weight_versions;
    uint16_t *low_versions;
    int *version_trace;
    uint64_t step;
};

// Negative rank selects the physical block for ordinary synchronous launches.
__device__ __forceinline__ int block_rank(const DeviceState &d) {
    return d.rank < 0 ? int(blockIdx.x) : d.rank;
}
