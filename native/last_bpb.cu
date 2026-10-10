// Read-only checkpoint evaluation on the archived, shared 8192-window panel.
// The checkpoint and destination follow the inline experiment.h settings.
#define main archived_trainer_main
#include "qwen.cu"
#undef main

namespace {
constexpr int Samples = 8192, Context = 4096, Batch = 16;
const fs::path Panel = "runs/qwen-hn1g/last4096-shared-8192";

std::vector<float> last_losses(Arena &arena, Blas &blas, Parameters &p,
                              const std::vector<uint32_t> &tokens) {
    arena.position = p.persistent;
    int rows = Batch * Context;
    std::vector<uint32_t> inputs(rows), targets(Batch);
    for (int b = 0; b < Batch; ++b) {
        std::copy_n(tokens.begin() + size_t(b) * (Context + 1), Context,
                    inputs.begin() + size_t(b) * Context);
        targets[b] = tokens[size_t(b) * (Context + 1) + Context];
    }
    auto ids = arena.alloc<uint32_t>(rows), target = arena.alloc<uint32_t>(Batch);
    cuda_check(cudaMemcpy(ids, inputs.data(), rows * 4, cudaMemcpyHostToDevice));
    cuda_check(cudaMemcpy(target, targets.data(), Batch * 4, cudaMemcpyHostToDevice));
    Graph graph(arena, blas, false);
    graph.context = Context; graph.batch = Batch; graph.adapters = true;
    T hidden = forward(graph, p, ids, rows);
    auto last = arena.alloc<float>(size_t(Batch) * Hidden);
    cuda_check(cudaMemcpy2DAsync(last, Hidden * 4,
        hidden->data + size_t(Context - 1) * Hidden, size_t(Context) * Hidden * 4,
        Hidden * 4, Batch, cudaMemcpyDeviceToDevice, cudaStreamPerThread));
    auto logits = arena.alloc<float>(size_t(Batch) * Vocab);
    auto losses = arena.alloc<float>(Batch);
    p.head(blas, logits, last, Batch);
    cross_entropy<<<Batch, 256>>>(logits, nullptr, target, losses, Vocab, 0);
    cuda_check(cudaGetLastError());
    cuda_check(cudaStreamSynchronize(cudaStreamPerThread));
    std::vector<float> result(Batch);
    cuda_check(cudaMemcpy(result.data(), losses, Batch * 4, cudaMemcpyDeviceToHost));
    for (float value : result)
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite last-token loss");
    return result;
}
}

int main() {
    try {
        static_assert(experiment::Context == Context && experiment::EffectiveBatch == Batch);
        static_assert(experiment::Rank == 16 && experiment::Alpha == 32);
        cuda_check(cudaSetDevice(experiment::Device));
        cudaDeviceProp device;
        cuda_check(cudaGetDeviceProperties(&device, experiment::Device));
        if (std::string(device.name) != "NVIDIA RTX PRO 6000 Blackwell Workstation Edition")
            throw std::runtime_error("Requires RTX PRO 6000 and CUDA_DEVICE_ORDER=PCI_BUS_ID");
        fs::path output = fs::path(experiment::Output) / "shared-8192";
        fs::create_directories(output);
        if (fs::exists(output / "last_nll.f32"))
            throw std::runtime_error("Evaluation scores already exist");
        std::ifstream paths(experiment::Assets);
        std::string model, manifest, train, eval, lengths;
        for (auto text : {&model, &manifest, &train, &eval, &lengths}) {
            std::getline(paths, *text);
            if (!text->empty() && text->back() == '\r') text->pop_back();
        }
        if (!paths) throw std::runtime_error("Missing model assets");
        std::ifstream input(Panel / "inputs.bin", std::ios::binary);
        uint64_t magic; uint32_t samples, context;
        read_binary(input, magic); read_binary(input, samples); read_binary(input, context);
        if (magic != 0x514e494e50555431ull || samples != Samples || context != Context)
            throw std::runtime_error("Wrong shared input panel");
        std::vector<uint32_t> windows(size_t(Samples) * (Context + 1));
        input.read(reinterpret_cast<char *>(windows.data()), windows.size() * 4);
        if (!input || input.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Invalid shared input length");
        std::vector<uint32_t> target_bytes(Samples);
        std::ifstream bytes(Panel / "target_bytes.u32", std::ios::binary);
        bytes.read(reinterpret_cast<char *>(target_bytes.data()), Samples * 4);
        if (!bytes || bytes.peek() != std::char_traits<char>::eof())
            throw std::runtime_error("Invalid target-byte table");
        uint64_t total_bytes = std::accumulate(target_bytes.begin(), target_bytes.end(), uint64_t(0));
        Arena arena(32 * GiB); Blas blas; blas.native_bf16 = true;
        Parameters p(arena, model, manifest, experiment::Rank, experiment::Alpha, experiment::Seed, true);
        blas.scratch_arena = &arena; blas.workspace(arena, GiB); p.persistent = arena.position;
        std::mt19937 unused_rng;
        int restored = restore(fs::path(experiment::Output) / "checkpoint.bin", p, Context, 1,
                               unused_rng, Batch, experiment::Optimizer);
        if (restored != experiment::Steps) throw std::runtime_error("Checkpoint is incomplete");
        std::cout << std::setprecision(17) << "{\"gpu\":\"" << device.name
                  << "\",\"context\":" << Context << ",\"samples\":" << Samples
                  << ",\"batch\":" << Batch << ",\"checkpoint_step\":" << restored
                  << ",\"learning_rate\":" << experiment::LearningRate << "}" << std::endl;
        std::ofstream scores(output / "last_nll.f32", std::ios::binary);
        auto begun = std::chrono::steady_clock::now(); double total_nll = 0;
        for (int index = 0; index < Samples; index += Batch) {
            auto first = windows.begin() + size_t(index) * (Context + 1);
            std::vector<uint32_t> batch(first, first + size_t(Batch) * (Context + 1));
            auto losses = last_losses(arena, blas, p, batch);
            if (index == 0) {
                Runner runner(arena, blas, p, lengths, Context, 1);
                Metrics full = runner.batch(batch, false);
                double observed = std::accumulate(losses.begin(), losses.end(), 0.);
                double error = std::abs(full.last_nll - observed) / Batch;
                if (error > 1e-4) throw std::runtime_error("Selected head disagrees with trainer loss");
                std::cout << "{\"trainer_head_mean_loss_error\":" << error << "}" << std::endl;
            }
            scores.write(reinterpret_cast<char *>(losses.data()), Batch * 4);
            if (!scores) throw std::runtime_error("Failed writing scores");
            total_nll += std::accumulate(losses.begin(), losses.end(), 0.);
            if ((index + Batch) % 256 == 0) {
                scores.flush();
                std::cout << "{\"samples_evaluated\":" << index + Batch
                          << ",\"elapsed_seconds\":" << seconds(begun) << "}" << std::endl;
            }
        }
        std::cout << "{\"evaluation_complete\":true,\"optimizer_updates\":0,\"last_bpb\":"
                  << total_nll / std::log(2.) / total_bytes << ",\"last_nll\":" << total_nll
                  << ",\"last_bytes\":" << total_bytes << ",\"seconds\":" << seconds(begun)
                  << ",\"peak_arena_gib\":" << double(arena.peak) / GiB << "}" << std::endl;
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl; return 1;
    }
}
