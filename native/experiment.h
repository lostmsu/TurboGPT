#pragma once

// Commit the implementation, then edit these constants inline for a run.
// build.ps1 embeds HEAD followed by the complete working-tree diff in TensorBoard.
namespace experiment {
constexpr int Context = 512, Batch = 16, EffectiveBatch = 128, Steps = 100;
constexpr double VramGiB = 12;
constexpr bool Evaluate = true, Sequential = false;
constexpr int EvalEvery = 10, EvalSamples = 16;
constexpr double CheckpointSeconds = 900;
constexpr const char *SequentialStarts = "";
constexpr const char *Assets = ".cache/native/paths.txt";
constexpr const char *Output = "runs/native-qwen";
constexpr const char *Dataset = "hn1g.txt";
constexpr unsigned long long DataBytes = 1'000'000'000;
}
