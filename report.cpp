// JSON output kept separate from the training loop.
#include "report.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

static void array(std::ostream &out, const std::vector<double> &values) {
    out << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i)
            out << ',';
        out << values[i];
    }
    out << ']';
}

void log_config(Engine *engine, const TrainingConfig &o, int steps, int64_t size) {
    std::cout << "{\"config\":{\"architecture\":\"swiglu48-rmsnorm\",\"depth\":" << o.model.depth
              << ",\"ctx\":" << o.model.context << ",\"batch\":" << o.model.batch
              << ",\"inflight\":" << o.model.inflight << ",\"chunk\":" << o.chunk
              << ",\"eval_every\":" << o.eval_every << ",\"optimizer\":\""
              << (o.model.muon ? "muon" : "mantissaadamw")
              << "\",\"parameters\":" << tg_parameter_count(engine)
              << ",\"blocks\":" << tg_blocks(engine)
              << ",\"shared_bytes\":" << tg_shared_bytes(engine) << ",\"steps\":" << steps
              << ",\"dataset_bytes\":" << size << ",\"schedule\":\"" << o.schedule
              << "\",\"pct_start\":" << o.onecycle.pct_start
              << ",\"div_factor\":" << o.onecycle.div_factor
              << ",\"final_div_factor\":" << o.onecycle.final_div_factor
              << ",\"cycle_momentum\":" << o.onecycle.cycle_momentum << "}}" << std::endl;
}

void log_step(const TrainingConfig &o, const Evaluation &final, int done, double train_seconds,
              float last_multiplier) {
    int64_t per_step = int64_t(o.model.batch) * o.model.context;
    std::cout << "{\"step\":" << done << ",\"tokens\":" << done * per_step
              << ",\"last_bpb\":" << final.last << ",\"last_bpb_se\":" << final.se
              << ",\"position_bpb\":";
    array(std::cout, final.positions);
    StepRates rates{last_multiplier, o.model.momentum};
    if (o.schedule == "onecycle")
        check_status(tg_onecycle_point(&o.onecycle, done - 1, &rates));
    std::cout << ",\"adam_lr\":" << o.model.learning_rate * rates.multiplier
              << ",\"muon_lr\":" << o.model.muon_lr * rates.multiplier << ",\"momentum\":"
              << (o.schedule == "onecycle" && o.onecycle.cycle_momentum ? rates.momentum
                                                                        : o.model.momentum)
              << ",\"train_seconds\":" << train_seconds
              << ",\"tokens_per_second\":" << done * per_step / train_seconds << "}" << std::endl;
}

static void report(std::ostream &out, const TrainingConfig &o, const RunReport &r) {
    out << std::setprecision(10)
        << "{\"architecture\":\"swiglu48-rmsnorm\",\"depth\":" << o.model.depth
        << ",\"ctx\":" << o.model.context << ",\"batch\":" << o.model.batch
        << ",\"inflight\":" << o.model.inflight << ",\"chunk\":" << o.chunk
        << ",\"eval_every\":" << o.eval_every << ",\"eval_batches\":" << o.eval_batches
        << ",\"final_batches\":" << o.final_batches << ",\"seed\":" << o.model.seed
        << ",\"optimizer\":\"" << (o.model.muon ? "muon" : "mantissaadamw")
        << "\",\"learning_rate\":" << o.model.learning_rate << ",\"muon_lr\":" << o.model.muon_lr
        << ",\"schedule\":\"" << o.schedule << "\",\"schedule_steps\":" << r.steps
        << ",\"pct_start\":" << o.onecycle.pct_start << ",\"div_factor\":" << o.onecycle.div_factor
        << ",\"final_div_factor\":" << o.onecycle.final_div_factor
        << ",\"cycle_momentum\":" << o.onecycle.cycle_momentum
        << ",\"base_momentum\":" << o.onecycle.base_momentum
        << ",\"max_momentum\":" << o.onecycle.max_momentum << ",\"requested_tokens\":" << o.tokens
        << ",\"tokens\":" << r.steps * (int64_t(o.model.batch) * o.model.context)
        << ",\"last_bpb\":" << r.final.last << ",\"last_bpb_se\":" << r.final.se
        << ",\"position_bpb\":";
    array(out, r.final.positions);
    out << ",\"batch_last_bpb\":";
    array(out, r.final.batch_last);
    out << ",\"initialization_seconds\":" << r.initialize_seconds
        << ",\"mean_staleness\":" << r.mean_staleness << ",\"max_staleness\":" << r.max_staleness
        << ",\"dataset_load_seconds\":" << r.load_seconds
        << ",\"training_seconds\":" << r.train_seconds
        << ",\"evaluation_seconds\":" << r.eval_seconds << ",\"process_seconds\":" << r.total
        << ",\"tokens_per_second\":"
        << r.steps * (int64_t(o.model.batch) * o.model.context) / r.train_seconds
        << ",\"evaluation\":\"sampled full stream; common target endpoints; no held-out "
           "split\",\"position_check_passed\":true}";
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
