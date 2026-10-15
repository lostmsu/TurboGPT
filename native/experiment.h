#pragma once

// Commit the implementation, then edit these constants inline for a run.
// build.ps1 embeds HEAD followed by the complete working-tree diff in TensorBoard.
namespace experiment {
constexpr int Device = 1, Rank = 16, WarmupSteps = 3;
constexpr unsigned Seed = 3407;
constexpr float Alpha = 32, LearningRate = 0.0030f;
constexpr const char *Optimizer = "muon", *Precision = "bf16", *Schedule = "warmup";
constexpr int Context = 4096, Batch = 4, EffectiveBatch = 16, Steps = 1240;
constexpr double VramGiB = 20;
constexpr bool Evaluate = false, Sequential = false;
constexpr int EvalEvery = 10, EvalSamples = 16;
constexpr double CheckpointSeconds = 900;
constexpr const char *SequentialStarts = "";
constexpr const char *Assets = ".cache/qwen-hn1g/paths.txt";
constexpr const char *Output = "runs/qwen-hn1g/lr-sweep-4096/lr-3e-3";
constexpr const char *Dataset = "hn1g.txt";
constexpr unsigned long long DataBytes = 1'000'000'000;
}
