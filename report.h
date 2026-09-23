#pragma once
#include "evaluation.h"

struct RunReport {
    int steps;
    Evaluation final;
    double initialize_seconds, load_seconds, train_seconds, eval_seconds, total;
    double mean_staleness;
    int max_staleness;
};
void log_config(Engine *engine, const TrainingConfig &options, int steps, int64_t size);
void log_step(const TrainingConfig &options, const Evaluation &evaluation, int done,
              double train_seconds, float last_multiplier);
void write_report(const TrainingConfig &options, const RunReport &report);
