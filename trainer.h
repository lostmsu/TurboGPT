#pragma once
#include "runtime.h"
#include "utils.h"
#include <string>

struct TrainingConfig {
    Config model;
    OneCycle onecycle;
    std::string schedule = "onecycle";
    std::string data, output, save, load;
    int64_t tokens = 100000000;
    int steps = 0, eval_every = 1000, eval_batches = 64, final_batches = 256, chunk = 1024;
    float final_lr = 1.f;
};

void train(TrainingConfig options, Clock::time_point process_start);
