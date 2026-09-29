#pragma once
#include "config.h"
#include "optim.h"
#include <cstdint>
#include <vector>

struct TrainingState {
    std::vector<float> weights, momentum, variance;
    OneCycle scheduler;
    uint64_t step = 0;
    double train_seconds = 0;
    std::vector<double> final_loss_sums;
    uint64_t final_loss_batches = 0;
};

TrainingState load_training_state(const std::string &path, const Config &config);
void save_training_state(const std::string &path, const Config &config, const TrainingState &state);
