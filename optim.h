#pragma once
#include <cmath>
#include <stdexcept>

// PyTorch-compatible two-phase cosine OneCycleLR. Rates are peak multipliers;
// inverse momentum applies to Muon momentum and AdamW beta1, not AdamW beta2.
struct OneCycle {
    int total_steps = 0, cycle_momentum = 1;
    double pct_start = 0.3, div_factor = 25, final_div_factor = 10000;
    double base_momentum = 0.85, max_momentum = 0.95;
};
struct StepRates {
    float multiplier, momentum;
};
inline StepRates onecycle_point(const OneCycle &c, int step) {
    if (c.total_steps < 1 || step < 0 || step >= c.total_steps ||
        !(c.pct_start > 0 && c.pct_start < 1) || !(c.div_factor > 0) || !(c.final_div_factor > 0) ||
        !(c.base_momentum >= 0 && c.base_momentum < 1) ||
        !(c.max_momentum >= c.base_momentum && c.max_momentum < 1) ||
        (c.cycle_momentum != 0 && c.cycle_momentum != 1))
        throw std::runtime_error("Invalid OneCycle schedule or step");
    double turn = c.pct_start * c.total_steps - 1;
    if (turn == 0)
        throw std::runtime_error("OneCycle warmup cannot contain exactly one step");
    bool warmup = step <= turn;
    double fraction = warmup ? step / turn : (step - turn) / (c.total_steps - 1 - turn);
    auto cosine = [fraction](double from, double to) {
        return to + (from - to) * 0.5 * (1 + std::cos(3.14159265358979323846 * fraction));
    };
    double initial = 1 / c.div_factor, final = initial / c.final_div_factor;
    return {float(warmup ? cosine(initial, 1) : cosine(1, final)),
            float(warmup ? cosine(c.max_momentum, c.base_momentum)
                         : cosine(c.base_momentum, c.max_momentum))};
}
