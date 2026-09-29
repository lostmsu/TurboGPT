#pragma once
#include "config.h"
#include <vector>

struct LayerParameters {
    int attention_norm, qkv, attention, mlp_norm, gate, up, down;
};
struct ParameterLayout {
    int embedding, position, final_norm, head, count;
    LayerParameters layers[8];
};
inline ParameterLayout parameter_layout(int depth, int context) {
    ParameterLayout p{};
    int n = 0;
    auto take = [&](int size) {
        int offset = n;
        n += size;
        return offset;
    };
    p.embedding = take(256 * 16);
    p.position = take(context * 16);
    for (int l = 0; l < depth; ++l) {
        auto &b = p.layers[l];
        b.attention_norm = take(16);
        b.qkv = take(48 * 16);
        b.attention = take(16 * 16);
        b.mlp_norm = take(16);
        // Adjacent gate/up matrices share one projection in the CUDA kernel,
        // but remain independent matrices for Muon orthogonalization.
        b.gate = take(48 * 16);
        b.up = take(48 * 16);
        b.down = take(16 * 48);
    }
    p.final_norm = take(16);
    p.head = take(256 * 16);
    p.count = n;
    return p;
}

// Parameter initialization and layout shared by host code.
std::vector<float> initialize_weights(const Config &config);
