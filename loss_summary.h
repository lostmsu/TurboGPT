#pragma once
#include <cstdint>
#include <vector>

class FinalLossAverage {
  public:
    FinalLossAverage(int total_steps, int positions, std::vector<double> sums = {},
                     uint64_t batches = 0);
    void add(int completed_step, const std::vector<float> &losses);
    std::vector<float> average() const;
    uint64_t batches() const;
    const std::vector<double> &sums() const;

  private:
    int first_step_;
    std::vector<double> sums_;
    uint64_t count_ = 0;
};
