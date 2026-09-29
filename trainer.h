#pragma once
#include "experiment.h"
#include "runtime.h"
#include "utils.h"
#include <string>

struct TrainingConfig {
    Config model = experiment_model();
    OneCycle onecycle = experiment_onecycle();
    std::string data, output, save, load, log_to = experiment_run_directory;
    bool logs_enabled = true;
    int64_t tokens = experiment_tokens;
    int chunk = experiment_chunk;
};

void train(TrainingConfig options, Clock::time_point process_start);
