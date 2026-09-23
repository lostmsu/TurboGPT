// Command-line entry point. All data loading and training are native C++.
#include "trainer.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static TrainingConfig arguments(int argc, char **argv) {
    TrainingConfig o;
    const char *user = std::getenv("USERPROFILE");
    o.data = std::string(user ? user : ".") + "/Downloads/Datasets/hn1g.txt";
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help") {
            std::cout
                << "turbogpt --data FILE --ctx 4|8 --depth 4|8 --optimizer mantissaadamw|muon\n"
                   "  --tokens N | --steps N; --batch N; --device N; --blocks N\n"
                   "  --lr X --muon-lr X --seed N --final-lr X --chunk 1..4096\n"
                   "  --inflight 1..256 (1: synchronous; larger: stale-gradient pipeline)\n"
                   "  --schedule onecycle|constant|cosine (default: onecycle; lr values are "
                   "peaks)\n"
                   "  --pct-start .3 --div-factor 25 --final-div-factor 10000\n"
                   "  --cycle-momentum 0|1 --base-momentum .85 --max-momentum .95\n"
                   "  --eval-every N --eval-batches N --final-batches N\n"
                   "  --output REPORT.json --save WEIGHTS.bin --load WEIGHTS.bin\n";
            std::exit(0);
        }
        if (i + 1 >= argc)
            throw std::runtime_error("Missing value for " + key);
        std::string value = argv[++i];
        if (key == "--data")
            o.data = value;
        else if (key == "--output")
            o.output = value;
        else if (key == "--save")
            o.save = value;
        else if (key == "--load")
            o.load = value;
        else if (key == "--ctx")
            o.model.context = std::stoi(value);
        else if (key == "--depth")
            o.model.depth = std::stoi(value);
        else if (key == "--batch")
            o.model.batch = std::stoi(value);
        else if (key == "--device")
            o.model.device = std::stoi(value);
        else if (key == "--blocks")
            o.model.blocks = std::stoi(value);
        else if (key == "--inflight")
            o.model.inflight = std::stoi(value);
        else if (key == "--seed")
            o.model.seed = std::stoull(value);
        else if (key == "--tokens")
            o.tokens = std::stoll(value);
        else if (key == "--steps")
            o.steps = std::stoi(value);
        else if (key == "--eval-every")
            o.eval_every = std::stoi(value);
        else if (key == "--eval-batches")
            o.eval_batches = std::stoi(value);
        else if (key == "--final-batches")
            o.final_batches = std::stoi(value);
        else if (key == "--chunk")
            o.chunk = std::stoi(value);
        else if (key == "--lr")
            o.model.learning_rate = std::stof(value);
        else if (key == "--muon-lr")
            o.model.muon_lr = std::stof(value);
        else if (key == "--final-lr")
            o.final_lr = std::stof(value);
        else if (key == "--schedule")
            o.schedule = value;
        else if (key == "--pct-start")
            o.onecycle.pct_start = std::stod(value);
        else if (key == "--div-factor")
            o.onecycle.div_factor = std::stod(value);
        else if (key == "--final-div-factor")
            o.onecycle.final_div_factor = std::stod(value);
        else if (key == "--cycle-momentum")
            o.onecycle.cycle_momentum = std::stoi(value);
        else if (key == "--base-momentum")
            o.onecycle.base_momentum = std::stod(value);
        else if (key == "--max-momentum")
            o.onecycle.max_momentum = std::stod(value);
        else if (key == "--clip")
            o.model.clip = std::stof(value);
        else if (key == "--optimizer") {
            if (value != "muon" && value != "mantissaadamw")
                throw std::runtime_error("Unknown optimizer " + value);
            o.model.muon = value == "muon";
        } else
            throw std::runtime_error("Unknown argument " + key);
    }
    if (o.tokens < 1 || o.steps < 0 || o.eval_every < 1 || o.eval_batches < 2 ||
        o.final_batches < 2 || o.chunk < 1 || o.chunk > 4096)
        throw std::runtime_error("Invalid token, step, evaluation, or chunk count");
    if (!(o.model.learning_rate > 0) || !(o.model.muon_lr > 0) ||
        !(o.final_lr > 0 && o.final_lr <= 1))
        throw std::runtime_error(
            "Learning rates must be positive; final-lr is a multiplier in (0,1]");
    if (o.schedule != "onecycle" && o.schedule != "constant" && o.schedule != "cosine")
        throw std::runtime_error("Schedule must be onecycle, constant, or cosine");
    if (o.final_lr != 1 && o.schedule != "cosine")
        throw std::runtime_error("--final-lr applies only to --schedule cosine");
    return o;
}

int main(int argc, char **argv) {
    auto started = Clock::now();
    try {
        train(arguments(argc, argv), started);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
