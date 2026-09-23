#pragma once
#include <cstdint>

// Width=16, heads=4, vocabulary=256. Depth and context are runtime values.
struct Config {
    int depth = 4, context = 4, batch = 2560, device = 0;
    int muon = 1, blocks = 0, ns_steps = 5;
    float learning_rate = 0.0006f, muon_lr = 0.02f;
    float beta1 = 0.9f, beta2 = 0.95f, momentum = 0.95f;
    float weight_decay = 0.1f, epsilon = 1e-8f, clip = 1.0f;
    uint64_t seed = 3407;
    int inflight = 1; // 1: synchronous; >1: ordered updates from versioned, stale gradients
};
