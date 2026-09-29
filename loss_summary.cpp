#include "loss_summary.h"
#include <algorithm>
#include <stdexcept>

FinalLossAverage::FinalLossAverage(int total_steps, int positions, std::vector<double> sums,
                                   uint64_t batches)
    : first_step_(std::max(0, total_steps - std::max(1, (total_steps + 99) / 100))),
      sums_(std::move(sums)), count_(batches) {
    if (sums_.empty())
        sums_.assign(positions, 0.0);
    if (int(sums_.size()) != positions)
        throw std::runtime_error("Final loss state position count mismatch");
}

void FinalLossAverage::add(int completed_step, const std::vector<float> &losses) {
    if (int(losses.size()) != int(sums_.size()))
        throw std::runtime_error("Loss position count mismatch");
    if (completed_step <= first_step_)
        return;
    for (size_t position = 0; position < losses.size(); ++position)
        sums_[position] += losses[position];
    ++count_;
}

std::vector<float> FinalLossAverage::average() const {
    if (!count_)
        return std::vector<float>(sums_.size(), 0.f);
    std::vector<float> result(sums_.size());
    for (size_t position = 0; position < sums_.size(); ++position)
        result[position] = float(sums_[position] / count_);
    return result;
}

uint64_t FinalLossAverage::batches() const {
    return count_;
}

const std::vector<double> &FinalLossAverage::sums() const {
    return sums_;
}
