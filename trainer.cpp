#include "trainer.h"
#include "dataset.h"
#include "evaluation.h"
#include "model.h"
#include "report.h"
#include <algorithm>
#include <climits>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>

void train(TrainingConfig o, Clock::time_point process_start) {
    std::cout << std::setprecision(9);
    auto initialization_start = Clock::now();
    std::unique_ptr<Engine, decltype(&tg_destroy)> engine(tg_create(&o.model), tg_destroy);
    if (!engine)
        throw std::runtime_error(tg_error());
    auto w = initialize_weights(o.model);
    if (!o.load.empty())
        load_weights(o.load, o.model, w);
    check_status(tg_set_weights(engine.get(), w.data()));
    int64_t per_step = int64_t(o.model.batch) * o.model.context;
    int64_t requested_steps = o.steps ? o.steps : (o.tokens + per_step - 1) / per_step;
    if (requested_steps > INT_MAX)
        throw std::runtime_error("Requested run exceeds the supported step count");
    int steps = int(requested_steps);
    o.onecycle.total_steps = steps;
    if (o.schedule == "onecycle")
        check_status(tg_set_onecycle(engine.get(), &o.onecycle));
    double initialize_seconds = seconds(initialization_start);
    auto load_start = Clock::now();
    int64_t size = std::filesystem::file_size(o.data);
    load_stream(engine.get(), o.data, size);
    double load_seconds = seconds(load_start);
    log_config(engine.get(), o, steps, size);
    double train_seconds = 0, eval_seconds = 0;
    float last_multiplier = 1;
    Evaluation final;
    for (int done = 0; done < steps;) {
        int boundary = std::min(steps, (done / o.eval_every + 1) * o.eval_every);
        auto segment = Clock::now();
        while (done < boundary) {
            int count = std::min(o.chunk, boundary - done);
            float progress = float(done) / std::max(1, steps - 1);
            float multiplier =
                o.schedule == "cosine"
                    ? o.final_lr + (1 - o.final_lr) * .5f *
                                       (1 + std::cos(3.14159265358979323846f * progress))
                    : 1.f;
            check_status(tg_train(engine.get(), count, multiplier));
            last_multiplier = multiplier;
            done += count;
        }
        check_status(tg_synchronize(engine.get()));
        train_seconds += seconds(segment);
        auto eval_start = Clock::now();
        final = evaluate(engine.get(), o, size, done == steps ? o.final_batches : o.eval_batches);
        eval_seconds += seconds(eval_start);
        log_step(o, final, done, train_seconds, last_multiplier);
    }
    if (!o.save.empty()) {
        check_status(tg_get_weights(engine.get(), w.data()));
        save_weights(o.save, o.model, w);
    }
    double mean_staleness = 0;
    int max_staleness = 0;
    check_status(tg_pipeline_stats(engine.get(), &mean_staleness, &max_staleness));
    engine.reset(); // include device teardown in total process timing
    double total = seconds(process_start);
    write_report(o, {steps, final, initialize_seconds, load_seconds, train_seconds, eval_seconds,
                     total, mean_staleness, max_staleness});
}
