// Path-only command line. Experiment settings are edited in experiment.h.
#include "trainer.h"
#include <cstdlib>
#include <iostream>
#include <stdexcept>

static TrainingConfig arguments(int argc, char **argv) {
    TrainingConfig options;
    const char *profile = std::getenv("USERPROFILE");
    options.data =
        std::string(profile ? profile : ".") + "/Downloads/Datasets/" + experiment_dataset_name;

    for (int index = 1; index < argc; ++index) {
        const std::string key = argv[index];
        if (key == "--help") {
            std::cout << "turbogpt --data FILE --load CHECKPOINT.pt --save CHECKPOINT.pt\n"
                         "  --log-to RUN_DIRECTORY --no-logs\n";
            std::exit(0);
        }
        if (key == "--no-logs") {
            options.logs_enabled = false;
            continue;
        }
        if (index + 1 >= argc)
            throw std::runtime_error("Missing value for " + key);
        const std::string value = argv[++index];
        if (key == "--data")
            options.data = value;
        else if (key == "--save")
            options.save = value;
        else if (key == "--load")
            options.load = value;
        else if (key == "--log-to") {
            options.log_to = value;
            options.logs_enabled = true;
        } else
            throw std::runtime_error("Unknown argument " + key);
    }
    return options;
}

int main(int argc, char **argv) {
    const Clock::time_point started = Clock::now();
    try {
        train(arguments(argc, argv), started);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
