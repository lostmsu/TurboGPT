#pragma once
#include "trainer.h"
#include "tensorboard.h"
#include <vector>

void log_tensorboard_info(TensorBoardLogger &logger, Engine *engine, const TrainingConfig &options,
                          int64_t data_bytes, int64_t optimizer_step);
void log_tensorboard_step(TensorBoardLogger &logger, Engine *engine, const TrainingConfig &options,
                          const std::vector<float> &training_losses, int done,
                          double train_seconds);
