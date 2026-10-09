#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

// Events accumulate in memory and reach the file when the buffer fills or on flush().
class TensorBoardLogger {
  public:
    explicit TensorBoardLogger(const std::filesystem::path &directory);
    ~TensorBoardLogger();
    void scalar(const std::string &tag, double value, int64_t step);
    void text(const std::string &tag, const std::string &value, int64_t step);
    void flush();

  private:
    // A scalar summary is constant per tag except for its 4 value bytes.
    struct ScalarSummary {
        std::vector<uint8_t> bytes;
        size_t value_offset;
    };
    void event(const std::vector<uint8_t> &summary, int64_t step);
    void record();
    void write_buffer();
    std::ofstream file;
    std::unordered_map<std::string, ScalarSummary> scalars;
    std::vector<uint8_t> event_bytes, buffer;
};
