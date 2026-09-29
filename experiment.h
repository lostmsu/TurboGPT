#pragma once
#include "config.h"
#include "optim.h"

// The experiment is part of the source revision. Edit these values for a run.
inline Config experiment_model() {
    Config model;
    model.depth = 4;
    model.context = 4;
    model.batch = 2880;
    model.device = 0;
    model.blocks = 0;
    model.ns_steps = 5;
    model.learning_rate = 0.0006f;
    model.muon_lr = 0.02f;
    model.beta1 = 0.9f;
    model.beta2 = 0.95f;
    model.momentum = 0.95f;
    model.weight_decay = 0.1f;
    model.epsilon = 1e-8f;
    model.clip = 1.0f;
    model.seed = 3407;
    model.inflight = 1;
    return model;
}

inline OneCycle experiment_onecycle() {
    OneCycle schedule;
    schedule.warmup_fraction = 0.30;
    schedule.initial_lr_fraction = 0.04;
    schedule.final_lr_fraction = 0.000004;
    schedule.low_momentum = 0.85;
    schedule.high_momentum = 0.95;
    return schedule;
}

constexpr int64_t experiment_tokens = 1'500'000'000;
constexpr int experiment_chunk = 1024;
constexpr const char *experiment_dataset_name = "hn1g.txt";
constexpr const char *experiment_run_directory = "runs/turbogpt";
constexpr int64_t maximum_tensorboard_reports = int64_t(8) * 1024 * 1024;
constexpr double checkpoint_interval_seconds = 900.0;
