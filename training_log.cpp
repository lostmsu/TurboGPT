#include "training_log.h"
#include "revision.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <sstream>

namespace {
std::string byte_size(int64_t bytes) {
    static const std::array<const char *, 6> units = {"bytes", "KB", "MB", "GB", "TB", "PB"};
    double value = double(bytes);
    size_t unit = 0;
    while (value >= 1024 && unit + 1 < units.size()) {
        value /= 1024;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(1) << value << ' ' << units[unit];
    return out.str();
}

std::string node() {
    const char *host = std::getenv("COMPUTERNAME");
    const char *devices = std::getenv("CUDA_VISIBLE_DEVICES");
    return std::string(host ? host : "localhost") + " (" + (devices ? devices : "all") + ")";
}

std::string model_description(const TrainingConfig &options) {
    std::ostringstream out;
    out << options.model.batch << "xswiglu48-rmsnorm-rope depth=" << options.model.depth
        << " context=" << options.model.context << " width=16 heads=4";
    return "```\n" + out.str() + "\n```";
}

std::string optimizer_description(const TrainingConfig &options) {
    std::ostringstream out;
    out << "Muon+MantissaAdamW"
        << "(lr=" << options.model.learning_rate << ", muon_lr=" << options.model.muon_lr
        << ", betas=(" << options.model.beta1 << ", " << options.model.beta2
        << "), momentum=" << options.model.momentum
        << ", weight_decay=" << options.model.weight_decay << ", epsilon=" << options.model.epsilon
        << ", clip=" << options.model.clip << ", schedule=onecycle)";
    return out.str();
}

double current_lr(const TrainingConfig &options, int done) {
    StepRates rates{};
    check_status(tg_onecycle_point(&options.onecycle, done - 1, &rates));
    return options.model.learning_rate * rates.multiplier;
}
} // namespace

void log_tensorboard_info(TensorBoardLogger &logger, Engine *engine, const TrainingConfig &options,
                          int64_t data_bytes, int64_t optimizer_step) {
    const int64_t model_bytes = int64_t(tg_parameter_count(engine)) * sizeof(float);
    logger.scalar("model_bytes", double(model_bytes), 0);
    logger.scalar("data_bytes", double(data_bytes), 0);
    logger.scalar("train_batch_size", options.model.batch, 0);
    logger.scalar("train_effective_batch_size", options.model.batch, 0);
    logger.text("revision", build_revision(), 0);
    logger.text("model_size", byte_size(model_bytes), 0);
    logger.text("dataset", std::filesystem::path(options.data).filename().string(), 0);
    logger.text("description", model_description(options), 0);
    logger.text("data_size", byte_size(data_bytes), 0);
    logger.text("node", node(), 0);
    logger.text("optimizer", optimizer_description(options), optimizer_step);
}

void log_tensorboard_step(TensorBoardLogger &logger, Engine *engine, const TrainingConfig &options,
                          const std::vector<float> &training_losses, int done,
                          double train_seconds) {
    const int64_t per_step = int64_t(options.model.batch) * options.model.context;
    const int64_t step = int64_t(done) * per_step;
    if (int(training_losses.size()) != options.model.context)
        throw std::runtime_error("Training loss history does not match the context");
    double loss_sum = 0;
    for (double value : training_losses)
        loss_sum += value;
    logger.scalar("loss", loss_sum / training_losses.size(), step);
    for (int base : {1, 2, 4, 16, 64, 256, 1024, 4096}) {
        const int position = base + 1;
        if (position < options.model.context)
            logger.scalar("loss_" + std::to_string(position), training_losses[position], step);
    }
    logger.scalar("lr", current_lr(options, done), step);
    logger.scalar("last_loss", training_losses.back(), step);
    logger.scalar("bits-per-byte", training_losses.back() / std::log(2.0), step);
    double used_gb = 0;
    check_status(tg_memory(engine, &used_gb));
    logger.scalar("mem", used_gb, step);
    logger.scalar("tokens_per_second",
                  train_seconds > 0 ? double(done) * per_step / train_seconds : 0, step);
}
