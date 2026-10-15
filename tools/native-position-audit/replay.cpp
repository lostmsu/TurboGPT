#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

int main() {
    namespace fs = std::filesystem;
    const fs::path output = ".cache/native-position-audit";
    fs::create_directories(output);
    constexpr std::uint64_t token_count = 284024229;
    constexpr int steps = 1240;
    for (int context : {4096, 1024, 256}) {
        std::mt19937 rng(3407);
        std::uniform_int_distribution<std::uint64_t> sample(0, token_count - context - 1);
        std::ofstream starts(output / ("starts-" + std::to_string(context) + ".u64"), std::ios::binary);
        const int effective_batch = 65536 / context;
        for (int step = 1; step <= steps; ++step)
            for (int lane = 0; lane < effective_batch; ++lane) {
                std::uint64_t position = sample(rng);
                starts.write(reinterpret_cast<const char *>(&position), sizeof(position));
            }
        starts.close();
        if (!starts) throw std::runtime_error("Cannot write replay positions");
        std::ofstream state(output / ("state-" + std::to_string(context) + ".txt"));
        state << rng;
        state.close();
        if (!state) throw std::runtime_error("Cannot write replay RNG state");
        std::cout << "Replayed context " << context << ": " << steps * effective_batch << " windows\n";
    }
}
