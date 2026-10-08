# turboGPT

Tiny byte-level GPT training in CUDA Rust. MIT.

Rust port of [TurboGPT](https://github.com/lostmsu/TurboGPT), with the original Git history
preserved so changes can be traced between the C++ and Rust codebases.

Host and device code share one crate. [cuda-oxide](https://github.com/NVlabs/cuda-oxide),
the SIMT track of [CUDA Rust](https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/),
compiles every `#[kernel]` to PTX and embeds it in the binary. Training is one persistent,
cooperative kernel per step chunk: BF16 tensor-core `mma.sync` and `ldmatrix` through inline
PTX, `cp.async` weight prefetch, grid-wide barriers, and a mantissa-preserving AdamW plus
Muon optimizer, all on the device. The Tile track (cutile-rs) cannot express this kernel: it
has no thread-level control, explicit shared memory or grid synchronization.

## Build

cuda-oxide supports Linux only. It needs CUDA 13, clang/libclang 21, rustup (the pinned
nightly in `rust-toolchain.toml` installs itself) and cargo-oxide:

```bash
cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
cargo oxide doctor
cargo oxide build
```

`cargo oxide build` emits PTX for sm_80, which the driver compiles once for newer GPUs and
caches; `--arch sm_120a` targets one architecture. Release builds only: unoptimized device
code currently fails libNVVM verification.

Windows uses a WSL 2 distro with the same tools (WSL's CUDA driver comes from Windows; install
the `cuda-toolkit-13-x` package from NVIDIA's `wsl-ubuntu` repository). `build.ps1` copies the
tree into `~/turboGPT-rs` there, builds, and optionally tests and runs:

```powershell
.\build.ps1 -Distro CudaRust -Test -Run -TrainingArgs '--data','/home/me/hn1g.txt'
```

## Run

```bash
./target/release/turbogpt --data hn1g.txt --log-to runs/ctx4
```

The run stores its checkpoint at `runs/ctx4/ctx4.pt`, containing model, optimizer,
scheduler, and trainer state. Use `--load CHECKPOINT.pt` to resume it. Checkpoints are
interchangeable with the CUDA C++ trainer.

`runs/ctx4/report.json` is derived from the log directory. Logs are TensorBoard-compatible:
one report per batch, capped at 8Mi reports, and flushed with periodic or final checkpoints.

Experiment settings live in `src/experiment.rs`.

## Result

- hn1g after 1.5G training tokens: **2.530 BPB** (mean of five runs).
- RTX PRO 6000 Blackwell, ctx4, batch 2880: 11.47 s of training (131M tokens/s), against
  11.59 s for the CUDA C++ trainer built with nvcc on the same GPU (mean of five alternating
  runs). The kernels take the same time at locked clocks. The GPU trains at its 330 W limit,
  so clock speed follows energy per step: spills and returning atomics cost throughput.

## Tests

The cdylib exposes the C interface the PyTorch oracle loads:

```bash
cargo oxide test -- --release --lib   # initialization and formatting units
python tests/verify.py                # model, gradients, causality, optimizers, schedule
python tests/pipeline.py              # versioned asynchronous updates
```

`TURBOGPT_LIBRARY` selects another build of the library.
