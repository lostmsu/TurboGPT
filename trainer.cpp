#include "trainer.h"
#include "checkpoint.h"
#include "dataset.h"
#include "loss_summary.h"
#include "model.h"
#include "report.h"
#include "sampling.h"
#include "tensorboard.h"
#include "training_log.h"
#include <algorithm>
#include <climits>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

namespace {
std::filesystem::path default_checkpoint_path(const std::filesystem::path &run_directory) {
    std::filesystem::path run_name = run_directory.filename();
    if (run_name.empty() || run_name == "." || run_name == "..")
        run_name = run_directory.parent_path().filename();
    if (run_name.empty() || run_name == "." || run_name == "..")
        run_name = "turbogpt";
    return run_directory / (run_name.string() + ".pt");
}

bool same_experiment(const TrainingConfig &options, const OneCycle &scheduler, int steps) {
    const OneCycle &expected = options.onecycle;
    return scheduler.total_steps == steps &&
           expected.warmup_fraction == scheduler.warmup_fraction &&
           expected.initial_lr_fraction == scheduler.initial_lr_fraction &&
           expected.final_lr_fraction == scheduler.final_lr_fraction &&
           expected.low_momentum == scheduler.low_momentum &&
           expected.high_momentum == scheduler.high_momentum;
}
} // namespace

void train(TrainingConfig options, Clock::time_point process_start) {
    if (options.save.empty())
        options.save = default_checkpoint_path(options.log_to).string();
    if (options.logs_enabled)
        options.output = (std::filesystem::path(options.log_to) / "report.json").string();
    std::cout << std::setprecision(9);

    const auto initialization_start = Clock::now();
    std::unique_ptr<Engine, decltype(&tg_destroy)> engine(tg_create(&options.model), tg_destroy);
    if (!engine)
        throw std::runtime_error(tg_error());

    TrainingState state;
    if (options.load.empty())
        state.weights = initialize_weights(options.model);
    else
        state = load_training_state(options.load, options.model);

    const int64_t tokens_per_step = int64_t(options.model.batch) * options.model.context;
    const int64_t requested_steps = (options.tokens + tokens_per_step - 1) / tokens_per_step;
    if (state.step > INT_MAX || requested_steps > INT_MAX)
        throw std::runtime_error("Run exceeds the supported step count");
    const int steps = int(requested_steps);
    options.onecycle.total_steps = steps;

    if (state.scheduler.total_steps != 0) {
        if (!same_experiment(options, state.scheduler, steps))
            throw std::runtime_error("Checkpoint does not match the source experiment");
    } else {
        state.scheduler = options.onecycle;
    }
    if (state.step >= steps)
        throw std::runtime_error("Checkpoint already reached the requested step count");

    const int parameter_count = tg_parameter_count(engine.get());
    state.momentum.resize(parameter_count);
    state.variance.resize(parameter_count);
    check_status(tg_set_weights(engine.get(), state.weights.data()));
    check_status(tg_set_onecycle(engine.get(), &options.onecycle));
    if (state.step > 0)
        check_status(tg_set_optimizer_state(engine.get(), state.momentum.data(),
                                            state.variance.data(), state.step));

    const double initialize_seconds = seconds(initialization_start);
    const auto load_start = Clock::now();
    const int64_t dataset_bytes = std::filesystem::file_size(options.data);
    load_stream(engine.get(), options.data, dataset_bytes);
    const double load_seconds = seconds(load_start);
    log_config(engine.get(), options, steps, dataset_bytes);

    std::unique_ptr<TensorBoardLogger> logger;
    if (options.logs_enabled)
        logger = std::make_unique<TensorBoardLogger>(options.log_to);
    if (logger)
        log_tensorboard_info(*logger, engine.get(), options, dataset_bytes,
                             int64_t(state.step) * tokens_per_step);

    double train_seconds = state.train_seconds;
    FinalLossAverage final_losses(steps, options.model.context, state.final_loss_sums,
                                  state.final_loss_batches);
    std::vector<float> latest_losses;
    const int report_stride = int((int64_t(steps) - 1) / maximum_tensorboard_reports + 1);
    const auto should_report = [steps, report_stride](int step) {
        return step == steps || step % report_stride == 0;
    };
    auto checkpoint_clock = Clock::now();

    const auto save_checkpoint = [&](int done) {
        state.step = uint64_t(done);
        state.train_seconds = train_seconds;
        state.scheduler = options.onecycle;
        state.final_loss_sums = final_losses.sums();
        state.final_loss_batches = final_losses.batches();
        check_status(tg_get_weights(engine.get(), state.weights.data()));
        uint64_t optimizer_step = 0;
        check_status(tg_get_optimizer_state(engine.get(), state.momentum.data(),
                                            state.variance.data(), &optimizer_step));
        if (optimizer_step != state.step)
            throw std::runtime_error("Optimizer step changed while saving checkpoint");
        save_training_state(options.save, options.model, state);
        if (logger)
            logger->flush();
        checkpoint_clock = Clock::now();
    };

    const auto record_chunk = [&](int first_step, const std::vector<float> &history) {
        const int count = int(history.size() / options.model.context);
        for (int offset = 0; offset < count; ++offset) {
            const int completed_step = first_step + offset + 1;
            const auto first = history.begin() + size_t(offset) * options.model.context;
            const std::vector<float> losses(first, first + options.model.context);
            final_losses.add(completed_step, losses);
            if (offset + 1 == count)
                latest_losses = losses;
            if (should_report(completed_step) && logger)
                log_tensorboard_step(*logger, engine.get(), options, losses, completed_step,
                                     train_seconds);
        }
        log_step(options, latest_losses, first_step + count, train_seconds);
    };
    // Training time runs from each launch until its losses arrive. Host logging for one
    // chunk overlaps the GPU training the next, so it only counts if the GPU waits for it.
    auto launch_start = Clock::now();
    const auto launch = [&](int from) {
        const int count = std::min(options.chunk, steps - from);
        launch_start = Clock::now();
        check_status(tg_train(engine.get(), count));
        return count;
    };

    int done = int(state.step);
    int running = launch(done);
    std::vector<float> history;
    while (running) {
        history.resize(size_t(running) * options.model.context);
        check_status(tg_get_training_losses(engine.get(), history.data(), running));
        train_seconds += seconds(launch_start);
        const int first_step = done;
        done += running;
        // A checkpoint reads device state at this chunk boundary, before the next launch.
        const bool checkpoint =
            done == steps || seconds(checkpoint_clock) >= checkpoint_interval_seconds;
        running = checkpoint ? 0 : launch(done);
        record_chunk(first_step, history);
        if (checkpoint) {
            save_checkpoint(done);
            if (done < steps)
                running = launch(done);
        }
    }

    double mean_staleness = 0;
    int max_staleness = 0;
    check_status(tg_pipeline_stats(engine.get(), &mean_staleness, &max_staleness));
    const auto sample_start = Clock::now();
    const std::string sample = final_sample(engine.get(), options);
    const double sample_seconds = seconds(sample_start);
    engine.reset(); // Include device teardown in process timing.
    write_report(options, {steps, final_losses.average(), sample, initialize_seconds, load_seconds,
                           train_seconds, sample_seconds, seconds(process_start), mean_staleness,
                           max_staleness});
}
