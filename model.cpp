#include "model.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

std::vector<float> initialize_weights(const Config &c) {
    ParameterLayout p = parameter_layout(c.depth, c.context);
    std::vector<float> w(p.count, 0.f);
    auto random = [&](int offset, int count, uint64_t key, float scale) {
        // Semantic seeds keep common initial matrices identical across ctx4/8.
        std::mt19937 generator(unsigned(hash_u64(c.seed + key)));
        std::normal_distribution<float> normal(0, scale);
        for (int i = 0; i < count; ++i)
            w[offset + i] = normal(generator);
    };
    random(p.embedding, 4096, 1, .02f);
    random(p.position, c.context * 16, 2, .02f);
    for (int l = 0; l < c.depth; ++l) {
        const LayerParameters &b = p.layers[l];
        std::fill_n(w.begin() + b.attention_norm, 16, 1.f);
        std::fill_n(w.begin() + b.mlp_norm, 16, 1.f);
        random(b.qkv, 768, 100 + l * 10, .02f);
        random(b.attention, 256, 101 + l * 10, .02f / std::sqrt(2.f * c.depth));
        random(b.gate, 768, 102 + l * 10, .02f);
        random(b.up, 768, 104 + l * 10, .02f);
        random(b.down, 768, 103 + l * 10, .02f / std::sqrt(2.f * c.depth));
    }
    std::fill_n(w.begin() + p.final_norm, 16, 1.f);
    random(p.head, 4096, 3, .02f);
    return w;
}

void save_weights(const std::string &path, const Config &c, const std::vector<float> &w) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
    std::ofstream file(path, std::ios::binary);
    // TG02: RMSNorm + bias-free SwiGLU48. TGPT checkpoints used GELU64.
    int header[] = {0x54473032, c.depth, c.context, int(w.size())};
    file.write(reinterpret_cast<const char *>(header), sizeof(header));
    file.write(reinterpret_cast<const char *>(w.data()), w.size() * 4);
    if (!file)
        throw std::runtime_error("Cannot save weights to " + path);
}

void load_weights(const std::string &path, const Config &config, std::vector<float> &weights) {
    std::ifstream f(path, std::ios::binary);
    int h[4]{};
    f.read(reinterpret_cast<char *>(h), sizeof(h));
    if (h[0] != 0x54473032 || h[1] != config.depth || h[2] != config.context ||
        h[3] != int(weights.size()))
        throw std::runtime_error(
            "Checkpoint architecture or shape mismatch (requires TG02 SwiGLU48)");
    f.read(reinterpret_cast<char *>(weights.data()), weights.size() * 4);
    if (!f)
        throw std::runtime_error("Truncated checkpoint");
}
