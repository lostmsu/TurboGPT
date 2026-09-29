#pragma once
#include <cmath>
#include <stdexcept>

// PyTorch-compatible OneCycleLR. Multipliers are fractions of the peak LR.
struct OneCycle {
    int total_steps = 0;
    double warmup_fraction = 0.30;
    double initial_lr_fraction = 0.04;
    double final_lr_fraction = 0.000004;
    double low_momentum = 0.85;
    double high_momentum = 0.95;
};
struct StepRates {
    float multiplier, momentum;
};

inline StepRates onecycle_point(const OneCycle &schedule, int step) {
    if (schedule.total_steps < 1 || step < 0 || step >= schedule.total_steps ||
        !(schedule.warmup_fraction > 0 && schedule.warmup_fraction < 1) ||
        !(schedule.initial_lr_fraction > 0 && schedule.initial_lr_fraction < 1) ||
        !(schedule.final_lr_fraction > 0 &&
          schedule.final_lr_fraction <= schedule.initial_lr_fraction) ||
        !(schedule.low_momentum >= 0 && schedule.low_momentum < 1) ||
        !(schedule.high_momentum >= schedule.low_momentum && schedule.high_momentum < 1))
        throw std::runtime_error("Invalid OneCycle schedule or step");

    const double warmup_steps = schedule.warmup_fraction * schedule.total_steps - 1;
    if (warmup_steps == 0)
        throw std::runtime_error("OneCycle warmup must span at least two steps");
    const bool warming_up = step <= warmup_steps;
    const double fraction = warming_up
                                ? step / warmup_steps
                                : (step - warmup_steps) / (schedule.total_steps - 1 - warmup_steps);
    const auto interpolate = [fraction](double from, double to) {
        return to + (from - to) * 0.5 * (1 + std::cos(3.14159265358979323846 * fraction));
    };
    return {
        float(warming_up ? interpolate(schedule.initial_lr_fraction, 1.0)
                         : interpolate(1.0, schedule.final_lr_fraction)),
        float(warming_up ? interpolate(schedule.high_momentum, schedule.low_momentum)
                         : interpolate(schedule.low_momentum, schedule.high_momentum)),
    };
}
