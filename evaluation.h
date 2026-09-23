#pragma once
#include "trainer.h"
#include <vector>

struct Evaluation {
    std::vector<double> positions, batch_last;
    double last = 0, se = 0;
};

Evaluation evaluate(Engine *engine, const TrainingConfig &options, int64_t size, int batches);
