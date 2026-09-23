#pragma once
#include "config.h"
#include "optim.h"
#include <stdexcept>

#ifdef _WIN32
#define TG_API extern "C" __declspec(dllexport)
#else
#define TG_API extern "C"
#endif

struct Engine;
TG_API const char *tg_error();
TG_API Engine *tg_create(const Config *config);
TG_API void tg_destroy(Engine *engine);
TG_API int tg_parameter_count(Engine *engine);
TG_API int tg_blocks(Engine *engine);
TG_API int tg_shared_bytes(Engine *engine);
TG_API int tg_set_weights(Engine *engine, const float *weights);
TG_API int tg_get_weights(Engine *engine, float *weights);
TG_API int tg_set_data(Engine *engine, const uint8_t *bytes, int64_t size);
TG_API int tg_train(Engine *engine, int steps, float lr_multiplier);
TG_API int tg_training_versions(Engine *engine, int *versions, int count);
TG_API int tg_pipeline_stats(Engine *engine, double *mean_staleness, int *max_staleness);
TG_API int tg_onecycle_point(const OneCycle *schedule, int step, StepRates *rates);
TG_API int tg_set_onecycle(Engine *engine, const OneCycle *schedule);
// Fixed endpoints are TARGET indices, shared between context-length comparisons.
TG_API int tg_evaluate(Engine *engine, const int64_t *endpoints, float *per_position);
// Fixed batches for independent PyTorch correctness / causality tests.
TG_API int tg_batch(Engine *engine, const uint8_t *x, const uint8_t *y, float *logits,
                    float *gradients, float *per_position);
TG_API int tg_optimizer_test(Engine *engine, const float *gradients, int steps);
TG_API int tg_synchronize(Engine *engine);

inline void check_status(int status) {
    if (status < 0)
        throw std::runtime_error(tg_error());
}
