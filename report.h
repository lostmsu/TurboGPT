#pragma once
#include "trainer.h"
#include <string>
#include <vector>

struct RunReport {
    int steps;
    std::vector<float> final_losses;
    std::string sample;
    double initialize_seconds, load_seconds, train_seconds, sample_seconds, total;
    double mean_staleness;
    int max_staleness;
};
void log_config(Engine *engine, const TrainingConfig &options, int steps, int64_t size);
void log_step(const TrainingConfig &options, const std::vector<float> &training_losses, int done,
              double train_seconds);
void write_report(const TrainingConfig &options, const RunReport &report);
