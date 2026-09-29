// JSON output kept separate from the training loop.
#include "report.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace {
void json_string(std::ostream &out, const std::string &value) {
    out << '"';
    static const char hex[] = "0123456789abcdef";
    for (unsigned char byte : value) {
        switch (byte) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (byte < 0x20 || byte >= 0x80)
                out << "\\u00" << hex[byte >> 4] << hex[byte & 15];
            else
                out << char(byte);
        }
    }
    out << '"';
}
} // namespace

void log_config(Engine *engine, const TrainingConfig &o, int steps, int64_t size) {
    std::cout << "{\"config\":{\"architecture\":\"swiglu48-rmsnorm-rope\",\"depth\":" << o.model.depth
              << ",\"ctx\":" << o.model.context << ",\"batch\":" << o.model.batch
              << ",\"inflight\":" << o.model.inflight << ",\"chunk\":" << o.chunk
              << ",\"optimizer\":\"muon\",\"parameters\":" << tg_parameter_count(engine)
              << ",\"blocks\":" << tg_blocks(engine)
              << ",\"shared_bytes\":" << tg_shared_bytes(engine) << ",\"steps\":" << steps
              << ",\"dataset_bytes\":" << size << ",\"schedule\":\"onecycle\""
              << ",\"warmup_fraction\":" << o.onecycle.warmup_fraction
              << ",\"initial_lr_fraction\":" << o.onecycle.initial_lr_fraction
              << ",\"final_lr_fraction\":" << o.onecycle.final_lr_fraction
              << ",\"low_momentum\":" << o.onecycle.low_momentum
              << ",\"high_momentum\":" << o.onecycle.high_momentum << "}}" << std::endl;
}

void log_step(const TrainingConfig &o, const std::vector<float> &training_losses, int done,
              double train_seconds) {
    int64_t per_step = int64_t(o.model.batch) * o.model.context;
    const double log2 = std::log(2.0);
    std::cout << "{\"step\":" << done << ",\"tokens\":" << done * per_step
              << ",\"last_bpb\":" << training_losses.back() / log2 << ",\"position_bpb\":[";
    for (size_t position = 0; position < training_losses.size(); ++position) {
        if (position)
            std::cout << ',';
        std::cout << training_losses[position] / log2;
    }
    std::cout << ']';
    StepRates rates{};
    check_status(tg_onecycle_point(&o.onecycle, done - 1, &rates));
    std::cout << ",\"adam_lr\":" << o.model.learning_rate * rates.multiplier
              << ",\"muon_lr\":" << o.model.muon_lr * rates.multiplier
              << ",\"momentum\":" << rates.momentum << ",\"train_seconds\":" << train_seconds
              << ",\"tokens_per_second\":" << done * per_step / train_seconds << "}" << std::endl;
}

static void report(std::ostream &out, const TrainingConfig &o, const RunReport &r) {
    const double log2 = std::log(2.0);
    const double final_loss = r.final_losses.back();
    out << std::setprecision(10)
        << "{\"architecture\":\"swiglu48-rmsnorm-rope\",\"depth\":" << o.model.depth
        << ",\"ctx\":" << o.model.context << ",\"batch\":" << o.model.batch
        << ",\"inflight\":" << o.model.inflight << ",\"chunk\":" << o.chunk
        << ",\"seed\":" << o.model.seed
        << ",\"optimizer\":\"muon\",\"learning_rate\":" << o.model.learning_rate
        << ",\"muon_lr\":" << o.model.muon_lr
        << ",\"schedule\":\"onecycle\",\"schedule_steps\":" << r.steps
        << ",\"warmup_fraction\":" << o.onecycle.warmup_fraction
        << ",\"initial_lr_fraction\":" << o.onecycle.initial_lr_fraction
        << ",\"final_lr_fraction\":" << o.onecycle.final_lr_fraction
        << ",\"low_momentum\":" << o.onecycle.low_momentum
        << ",\"high_momentum\":" << o.onecycle.high_momentum << ",\"requested_tokens\":" << o.tokens
        << ",\"tokens\":" << r.steps * (int64_t(o.model.batch) * o.model.context)
        << ",\"final_loss\":" << final_loss << ",\"final_bpb\":" << final_loss / log2
        << ",\"position_bpb\":[";
    for (size_t position = 0; position < r.final_losses.size(); ++position) {
        if (position)
            out << ',';
        out << r.final_losses[position] / log2;
    }
    out << ']';
    out << ",\"sample\":";
    json_string(out, r.sample);
    out << ",\"initialization_seconds\":" << r.initialize_seconds
        << ",\"mean_staleness\":" << r.mean_staleness << ",\"max_staleness\":" << r.max_staleness
        << ",\"dataset_load_seconds\":" << r.load_seconds
        << ",\"training_seconds\":" << r.train_seconds << ",\"sample_seconds\":" << r.sample_seconds
        << ",\"process_seconds\":" << r.total << ",\"tokens_per_second\":"
        << r.steps * (int64_t(o.model.batch) * o.model.context) / r.train_seconds
        << ",\"sample_temperature\":1.0}";
}

void write_report(const TrainingConfig &o, const RunReport &r) {
    std::cout << "{\"summary\":";
    report(std::cout, o, r);
    std::cout << "}" << std::endl;
    if (!o.output.empty()) {
        auto parent = std::filesystem::path(o.output).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
        std::ofstream file(o.output);
        report(file, o, r);
        file << '\n';
        if (!file)
            throw std::runtime_error("Cannot write report");
    }
}
