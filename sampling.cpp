#include "sampling.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

std::string final_sample(Engine *engine, const TrainingConfig &options) {
    constexpr int GeneratedBytes = 80, TopK = 10;
    constexpr double Temperature = 1.0;
    const std::string prompt = "O God, O God!";
    std::vector<uint8_t> sequence(prompt.begin(), prompt.end());
    std::mt19937 random(options.model.seed);
    std::vector<float> logits(256);
    std::array<int, 256> tokens{};
    std::iota(tokens.begin(), tokens.end(), 0);
    for (int step = 0; step < GeneratedBytes; ++step) {
        std::vector<uint8_t> context(options.model.context, 0);
        const size_t available = std::min<size_t>(sequence.size(), context.size());
        std::copy(sequence.end() - available, sequence.end(), context.end() - available);
        check_status(tg_predict(engine, context.data(), logits.data()));
        std::partial_sort(tokens.begin(), tokens.begin() + TopK, tokens.end(),
                          [&](int left, int right) { return logits[left] > logits[right]; });
        std::vector<double> probabilities(TopK);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int token = 0; token < TopK; ++token)
            maximum = std::max(maximum, double(logits[tokens[token]]));
        double sum = 0;
        for (int token = 0; token < TopK; ++token) {
            probabilities[token] = std::exp((logits[tokens[token]] - maximum) / Temperature);
            sum += probabilities[token];
        }
        for (double &probability : probabilities)
            probability /= sum;
        std::discrete_distribution<int> choose(probabilities.begin(), probabilities.end());
        sequence.push_back(uint8_t(tokens[choose(random)]));
    }
    return std::string(sequence.begin(), sequence.end());
}
