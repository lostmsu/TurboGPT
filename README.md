# Qwen-Tuning

Native **CUDA C++** continued pretraining of
[Qwen3.5-0.8B-Base](https://huggingface.co/Qwen/Qwen3.5-0.8B-Base) on the raw
[hn1g](https://huggingface.co/datasets/lostmsu/hn1g) Hacker News stream.
Forward, backward, LoRA, cross-entropy, Muon and AdamW run in CUDA/cuBLAS.
Python prepares assets and runs independent reference validation; the training executable has no
Python, PyTorch, Transformers, PEFT or bitsandbytes dependency.

The trainer implements Qwen's complete 24-layer text backbone: 18 Gated DeltaNet
layers, six grouped-query attention layers, gated attention outputs, partial
rotary embeddings, offset RMSNorm, SwiGLU and the tied embedding/output head.
The vision backbone is omitted. The TensorBoard writer, OneCycle schedule,
Muon equations and mantissa-preserving AdamW representation come from the original
CUDA C++ TurboGPT codebase. The previous Rust experiment is preserved in `src/`
and [README-turbogpt.md](README-turbogpt.md), with `build-rust.ps1` as its build
entry point. The superseded PyTorch port is preserved in
`validation/pytorch-reference/` for comparison, with its own dependency lockfile
and tests. It is independent of the native trainer and root Python environment.
The original position-loss and paired-evaluation analysis helpers are preserved
in `tools/native-position-audit/`; their generated files remain under `.cache/`
and `runs/`. These directories, including result summaries, remain untracked.

## Prepare the assets

Use Python 3.12/3.13 and `uv`. These dependencies contain no neural framework.

```powershell
uv sync --locked
uv run --locked python tools/native_assets.py
```

The full model snapshot is pinned to
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`, in the standard local Hugging Face
cache at **`.cache/huggingface/hub/`**. Preparation emits a safetensors byte-offset
registry, token byte-length table and native path manifest under `.cache/native/`.
It reads the model without a tensor-framework loader. Add `--offline` to prohibit
downloads, `--data PATH` to specify a local dataset, or `--tokens CACHE_DIRECTORY`
to reuse a previously prepared and verified token cache.

Data discovery checks `./hn1g.txt`, then `~/Downloads/Datasets/hn1g.txt`, then
downloads `hn1g.txt` from the dataset repository. The published file is
1,000,000,000 bytes, SHA-256:
`000beb89de92941829a2f64ee3f0f500de92c98959dd47e97b9a7347b9b285b9`.
Preparation preserves whitespace, record markup and multiline bodies. The last
1% of bytes forms a chronological holdout; the split advances to a record
header. Train and validation are tokenized separately, without added special
tokens. Invalid UTF-8 is replaced during decoding.
Qwen's tokenizer also applies NFC Unicode normalization.

Bounded tokenization writes uint32 caches under `.cache/tokens/`; cache manifests
record source hash, tokenizer revision, split and token counts. The complete
local corpus has **281,186,047 train tokens** and **2,838,182 validation tokens**,
with the byte split at 990,000,029. To reuse this existing cache:

```powershell
uv run --locked python tools/native_assets.py --offline --tokens .cache/tokens/1b1b1421ba526d0e875b
```

## Build and train

The six equal-token experiments use the **full corpus**, with no held-out split
or evaluation. Their asset manifest joins both historical token caches in source
order: **284,024,229 tokens** covering all 1,000,000,000 source bytes.
The earlier validation experiments below retain their historical holdout.

After committing the common implementation, run:

```powershell
uv sync --locked --extra experiments
uv run --locked --extra experiments python tools/run_hn1g_suite.py
```

The controller runs `runs/qwen-hn1g/v-4096`, `v-1024`, `v-256`, followed by
`v-seq-4096`, `v-seq-1024`, `v-seq-256`. Each starts from the original checkpoint
with seed 3407, BF16 LoRA rank 16 / alpha 32, Muon LR 0.001 and three warmup
updates. All six process **81,264,640 predicted tokens** in **1,240 updates** of
65,536 tokens. Context 4096 uses physical/effective batches 4/16 with a 20 GiB
arena; 1024 uses 8/64 and 256 uses 32/256 with 12 GiB arenas. The token budget
corresponds to roughly two hours at the measured 4096-context throughput.

Each variation changes `native/experiment.h` inline and remains uncommitted.
The build embeds the committed SHA plus the full working-tree diff; each run
archives that revision, the executable, source, SHA-256 hashes, compiler/GPU
details, asset manifest and command. TensorBoard uses the original ctx4 tags:
`loss`, `loss_<position>`, `lr`, `last_loss`, `bits-per-byte`, `mem`,
`tokens_per_second`, the four model/data/batch metadata scalars, and seven text
summaries including `revision`. The horizontal axis is **processed tokens**.
`bits-per-byte` is the true last-token NLL / UTF-8-byte ratio; `mem` is allocated
GiB. Additional `train/` tags preserve the Qwen diagnostics.

Sequential lanes start at `actual_file_bytes * j / effective_batch`, aligned
forward to a token boundary, and advance exactly `context` tokens per update.
The byte spacing uses the actual 1,000,000,000-byte file, rather than 1 GiB.
Qwen's NFC-normalized stream is 999,999,998 bytes; alignment uses that stream's
byte lengths, with the actual anchor and small adjustment recorded per lane.
Both preparation and the native trainer verify that every lane's complete run,
including the final next-token target, stays before the next start or corpus end.
There is no wraparound.

Durable adapter/optimizer/RNG checkpoints are saved every 900 seconds and at
completion. Rerun the same controller command after a restart to resume each
unfinished run with its archived executable. Updates past the last durable
checkpoint are archived and removed from the active JSON/TensorBoard histories
before replay, keeping the logs consistent. Completed runs are verified and
skipped. `status.json` records live progress; `summary.json`, `summary.csv`,
`summary.md` and `loss.png` update as runs finish. The reported average last bpb
uses the final **4,194,304 processed tokens**, or 64 updates, for every run.
The primary statistic is the arithmetic mean of those per-update training
last-bpb values; a pooled byte-weighted ratio is also recorded. TensorBoard's
package reader verifies the tags, token axis, scalar values and revision text.

For a short end-to-end smoke suite in a separate directory:

```powershell
uv run --locked --extra experiments python tools/run_hn1g_suite.py --root runs/qwen-hn1g-smoke --steps 2 --window-tokens 131072
```

On Windows install MSVC C++ tools and CUDA 13.4. `QWEN_CUDA` can override the
toolkit path. The script initializes the MSVC environment and builds directly
with `nvcc`; WSL and Rust are unnecessary for Qwen training.

```powershell
# RTX PRO 6000 Blackwell, device 1 on this machine.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--self-test')

# BF16 training, after generating the shared subset below.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--steps','60','--eval-every','20','--eval-input','runs/bf16-accuracy-fp32-reference/inputs.bin','--output','runs/native-bf16-hn1g')
```

The default build includes architectures 86 and 120. Use 86 for an RTX 3090 or
89 for an RTX 4090. CUDA device numbering uses PCI bus order, matching
`nvidia-smi` on this machine. GPU 1 is the RTX PRO 6000; GPU 0 is a busy RTX 5060 Ti.

A CMake build is also available:

```sh
cmake -S . -B build/native -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build/native --config Release
```

On Windows, run CMake from a Visual Studio developer shell so the linker can
find the Windows SDK libraries. `build.ps1` initializes that environment itself.

`qwen-tune --help` lists all options. Direct execution requires CUDA's runtime
DLL directory on PATH on Windows; `build.ps1 -Run` sets it automatically.

## Configuration and memory

| Setting | Default |
| --- | --- |
| Adapter | LoRA rank 16, alpha 32, no dropout |
| Targets | All 186 text projections; 10,822,656 trainable parameters |
| Base | Frozen matrices stored directly in the checkpoint's BF16 format; scalar/norm/convolution weights FP32 |
| Arithmetic | BF16 GEMM operands with FP32 accumulation/output; FP32 pointwise operations, recurrent state and optimizer master weights |
| Context / physical batch | 512 tokens / 16 independent sequences |
| Total optimizer batch | 128 sequences at context 512; scales to 16 at context 4096, preserving 65,536 predicted tokens per update |
| Accumulation | Automatic: effective sequences divided by physical batch; eight microbatches at the defaults |
| Objective | Exact next-token cross-entropy, vocabulary chunks of up to 2,048 rows |
| Optimizer | Muon, LR 1e-3, momentum 0.95, five Newton-Schulz iterations, decay 0.01 |
| Optimizer alternatives | `--optimizer mantissa-adamw` or legacy `adamw`; default LR 1e-4, betas 0.9/0.999 |
| Activation memory | Per-layer recomputation and short recurrent state checkpoints |
| Execution | Cached CUDA graphs on a per-thread stream; 1,024 MiB cuBLAS workspace inside the arena |
| Clipping / warmup | Global gradient norm 1; three updates |
| Sampling | Random contiguous training blocks, with replacement |
| Validation | 16 fixed blocks evenly spread through the holdout |
| Arena | 12 GiB; allocations fail before exceeding this cap |
| Stop | 100 updates, or `--seconds N` when supplied |

Unquantized LoRA fits the requested memory envelope for this small model and
keeps the pretrained weight values. Four-bit quantization is unnecessary here.
DeltaNet distributes state tiles across warps, replays four-token chunks in
registers and combines warp derivatives before writing to VRAM. Layer
recomputation avoids retaining all 24 layers' activations at once. The verified
512-token, batch-16 BF16 configuration peaks at **9.36 GiB of arena storage used**;
the arena reserves 12 GiB. Driver/context/cuBLAS allocations are
outside that arena; the observed total GPU-memory increment is approximately
12.5 GiB, leaving substantial room on a 24 GiB card. This run was executed on the
96 GiB RTX PRO 6000 under the same fixed cap.

Set `--batch` (alias `--virtual-batch`) to the physical number of independent
sequences processed together. It must divide the context's effective optimizer batch.
The trainer automatically accumulates gradients before a single clipping and
optimizer step: batch 16 accumulates eight times, 32 four times, 64 twice,
and 128 once at context 512. There is no total-batch CLI option. At a fixed context, changing
the physical batch preserves the number of predicted tokens and random windows
per optimizer update. Floating-point reduction order can change slightly.
Checkpoint resume permits a new physical batch when the accumulated total
matches the checkpoint.

```powershell
# Fastest measured configuration: 16 physical sequences, eight microbatches.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--batch','16','--vram-gib','12','--output','runs/qwen-24g')

# Larger physical batches can use the 96 GiB card; total remains 128.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--batch','128','--vram-gib','64','--output','runs/qwen-96g')
```

The arena reserves its requested capacity upfront; choose a cap that fits the
currently free VRAM with at least 0.5 GiB outside it for the CUDA context.
The token budget and scaling policy are defined in `native/qwen.cu`. Historical runs used a
total of 32; explicit `--accum 1 --batch 32` remains supported for reproducing
their settings and restoring those checkpoints. It is also used for small
independent validation runs. The CLI accepts contexts 8..4096. Effective sequences
are `65536 / max(context, 512)`, rounded down. Contexts below 512 retain the
historical total of 128; contexts that do not divide 65,536 round down the number
of complete sequences. The default physical batch targets at most 8,192 tokens
and is reduced to a divisor of the effective batch. Checkpoint resume requires
the same context and effective batch, but permits changing the physical batch.

| Context | Default physical batch | Effective sequences | Accumulation | Predicted tokens/update |
| --- | --- | --- | --- | --- |
| 512 | 16 | 128 | 8 | 65,536 |
| 1,024 | 8 | 64 | 8 | 65,536 |
| 2,048 | 4 | 32 | 8 | 65,536 |
| 4,096 | 2 | 16 | 8 | 65,536 |

At 4096 tokens, `--batch 4`, `8` and `16` accumulate four, two and one times,
respectively. Validation and exported inference now recycle layer scratch
storage, avoiding retention of all six attention matrices. Training still uses
per-layer recomputation; the attention implementation materializes a full
causal score matrix within each attention layer, so longer contexts add work.

```powershell
# Long-context default: physical batch 2, effective batch 16, 12 GiB arena.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--context','4096','--steps','60','--output','runs/qwen-4096')

# More physical sequences on the 96 GiB RTX PRO 6000; effective stays 16.
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--context','4096','--batch','16','--vram-gib','64','--output','runs/qwen-4096-b16')
```

The 4096-token sweep on driver **617.42**, with the same 330 W power cap,
keeps 16 sequences / 65,536 tokens per update. Six updates use identical hn1g
windows, and timing excludes the first two updates:

| Physical batch | Accumulation | Tokens/sec | Peak working storage | Arena reserved |
| --- | --- | --- | --- | --- |
| 2 | 8 | 9,629 | 9.36 GiB | 12 GiB |
| 4 | 4 | 10,547 | 16.07 GiB | 20 GiB |
| 8 | 2 | 10,497 | 29.49 GiB | 34 GiB |
| 16 | 1 | 10,450 | 56.34 GiB | 64 GiB |

Batch 4 was slightly fastest and fits under 24 GiB including runtime overhead.
Use `--context 4096 --batch 4 --vram-gib 20` for this configuration. Batch 2
remains the lower-memory default. The larger batches can use the 96 GiB card;
they did not improve throughput in this sweep. Reports, target-count and batch
geometry checks, source snapshot and executable are in
`runs/context4096-driver61742/`. The earlier driver-616.92 sweep is retained in
`runs/context4096-batch-sweep/`; its batch-16 run was interrupted by the PC restart.
The matching pip PyTorch BF16/FLA/SDPA batch-4 benchmark reaches **12,616
tokens/sec**, leaving a **16.4% native throughput gap** at 4096 tokens. This
comparison uses the same six updates and excludes the first two; the longer
512-token benchmark below has a different context and driver.

Long-context inference validation uses **64 identical random holdout windows**.
The installed pip Transformers IEEE reference records original-model last bpb
**0.960215713**; native corrected-TF32 records **0.960213411** (difference
**0.00000230**), with strict hidden-state/logit checks passing and all 64 top
predictions matching. Zero adapters preserve native hidden/logit bits and Python
outputs exactly. The BF16 training arithmetic records **0.959804002**, last-bpb
difference **0.000411712**, logit RMSE **0.01047**, and 64/64 top predictions.
Its worst hidden-row relative error is **6.19%**, exceeding the existing 3%
cross-precision bound. This is retained as an accuracy measurement, not a strict
parity pass, in `runs/context4096-accuracy-bf16/`. The strict report is in
`runs/context4096-parity-strict/`; both use the subset in
`runs/context4096-validation-subset/`.

The native BF16/Muon loss-watch run completes **60 updates / 3,932,160 predicted
tokens** at context 4096, physical batch 4, effective batch 16. It evaluates the
same 64 holdout windows every 20 updates. Validation loss falls **2.7864 →
2.6690**, all-token bpb **1.13808 → 1.09016**, last-token loss **2.51561 →
2.35746**, and last bpb **0.959804 → 0.899463**. Both all-token loss and last bpb
improve at all three recorded evaluations. The run takes **394.8 seconds**
including evaluation/checkpoint writes, with **11,267 steady training tokens/sec**
on its random windows; this throughput is separate from the shared-window
PyTorch comparison above. Working storage peaks at **16.07 GiB**, with a 20 GiB
arena. Observed total device-memory use is **20.57 GiB**, including runtime and
background graphics allocations. Reports, checkpoints, TensorBoard, CSV and
PNG/SVG loss plots are in `runs/native-bf16-4096-hn1g/`. Restoring with physical
batch 2 / accumulation 8 reproduces every adapter/optimizer array and the
evaluation exactly; see `runs/native-bf16-4096-restore/`.

Nsight Compute hardware counters for one complete 4096-token, batch-4 update
measure **19.47% tensor pipeline activity**, **26.79% SM throughput** and
**52.88% DRAM throughput** relative to sustained peak. The profiled range takes
6.03 seconds; clock control is disabled and the mean GPC clock is 1.464 GHz.
These are hardware counters, distinct from the GPU-busy statistic. They do not
show near-full GPU or tensor-core utilization. Raw reports, configuration and
provenance are in `runs/context4096-profile/`.

```powershell
validation/.venv/Scripts/python.exe validation/compare.py --context 4096 --samples 64 --precision tf32x2 --output runs/qwen-4096-parity-new
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--context','4096','--batch','4','--vram-gib','20','--steps','60','--eval-every','20','--eval-input','runs/qwen-4096-parity-new/inputs.bin','--output','runs/qwen-4096-watch-new')
```

The default `--precision bf16` loads frozen matrix bytes directly into BF16 GPU
storage and passes actual BF16 operands to cuBLAS. Frozen matrices have no FP32
duplicate. Projection inputs and gradients are converted once and shared across
their LoRA GEMMs. SwiGLU/gating kernels also write packed operands and gradients,
and cross-entropy writes its BF16 gradient directly. FP32 activation/state and
master-weight storage preserves sensitive reductions and small optimizer updates.
This is mixed-precision LoRA; the BF16 checkpoint never required all-FP32 training.

The optional `--precision tf32x2` explicitly splits a projection input into its
rounded TF32 value and residual, then multiplies both components by the frozen
BF16 weight. BF16 weights are represented exactly in TF32. Adding the second
product recovers most of the input precision discarded by plain TF32. Both
products accumulate in FP32. Attention and adapter products use conventional
FP32 because neither operand has the frozen BF16 guarantee. Scratch storage is
inside the arena and released after each GEMM, including during graph capture.
The GEMM self-test covers batching, both transposes, either frozen operand,
nonzero alpha/beta and scratch reclamation. `--precision fp32` provides the
strict reference path; `--precision tf32` selects the faster approximate path.
Plain TF32 matched all 128 top predictions and last bpb closely but failed the
stricter hidden-state/logit tolerance on several outliers. CUDA
13.4's `--precision bf16x9` emulation is supported on compute capabilities 10.0
and 10.3; on this RTX's 12.0 architecture it falls back to ordinary FP32, as the
hardware counters and kernel timeline confirm. It provides no speed benefit
here. See the
[cuBLAS emulation documentation](https://docs.nvidia.com/cuda/cublas/index.html#floating-point-emulation).

## BF16 performance

The final native CUDA C++ build reaches **12,986 tokens/sec**, versus **13,261**
for pip PyTorch BF16 with FLA and SDPA: a **2.1% throughput gap**. Both use physical
batch 16, total batch 128, context 512, identical hn1g windows and adapter
initialization, all 186 projections, Muon and exact full-vocabulary loss. These
are six-update measurements excluding the first two updates. The RTX PRO 6000
retained its 330 W limit. PyTorch batch 32 measured 12,568 tokens/sec.

A longer comparison runs **20 updates and excludes the first ten** to account
for warmup and sustained-load effects: native **12,582 tokens/sec**, pip
PyTorch **13,079**, a **3.8% gap**. Both consume identical windows and use the
same batch 16 / effective batch 128. Reports and plots are in
`runs/bf16-performance-warm/`. Reproduce with the commands above adding
`--steps 20 --warmup 10` and using a fresh common output directory.

The checkpoint matrices remain BF16; this implementation deliberately keeps
pointwise operations and DeltaNet state FP32, while PyTorch's benchmark uses its
BF16 model and FLA/SDPA kernels. Their arithmetic and outputs are not bit-equivalent.
The native peak is 9.36 GiB of working arena storage, reserved capacity 12 GiB;
PyTorch batch 16 peaks at 9.60 GiB allocated and 10.97 GiB reserved. Driver/context
memory is additional. Increasing native physical batch to 32 or 64 was slower
in the preceding tuning sweep, despite reducing accumulation.
The final build measures 12,219 tokens/sec at batch 32 (16.07 GiB used,
22 GiB arena) and 11,642 at batch 128 (56.35 GiB used, 64 GiB arena).
The observed default-training GPU-memory increment is 12.55 GiB including
driver/context overhead, above the 12 GiB reserved arena.

`runs/bf16-performance-final/` holds shared inputs, per-update reports, executable
and source hashes. `runs/bf16-performance/experiments/` preserves checkpoint-spacing
and operand-cache experiments. Eight-token state replay reduced storage further
but used 255 registers/thread and was slightly slower than four-token replay
(128 registers/thread); four was retained.

Reproduce the controlled comparison with the separate reference environment:

```powershell
uv run --locked python validation/benchmark_training.py --backend native --precision bf16 --batch 16 --vram-gib 12 --output runs/bf16-comparison-new
validation/.venv/Scripts/python.exe validation/benchmark_training.py --backend torch --precision bf16 --delta fla --batch 16 --output runs/bf16-comparison-new
uv run --with matplotlib python tools/summarize_benchmarks.py runs/bf16-comparison-new
```

The longer native run in `runs/native-bf16-hn1g/` completes **60 updates / 3,932,160
predicted tokens** in 334.5 seconds including evaluation/checkpoint writes
(315.9 seconds training, **12,446 steady tokens/sec**). It keeps the effective
batch at 128 and evaluates the same 128 random holdout windows every 20 updates.
Validation loss decreases **3.1011 → 2.7376**, all-token bpb **1.2674 → 1.1188**,
last-token loss **3.0631 → 2.9393**, and last bpb **1.18584 → 1.13793**.
Peak working storage stays 9.36 GiB. Restoring its checkpoint reproduces every
adapter/optimizer array and all evaluation metrics exactly. Reports, TensorBoard
events, provenance, CSV and PNG/SVG loss plots are saved with the run.
Last-position metrics reached their lowest observed values at update 40 and
rose at 60; all-token validation loss continued to decrease.

## Historical corrected-TF32 Muon run

The default native configuration completed **100 updates / 1,638,400 training
tokens** in 281.8 seconds including validation.
Training-only throughput was **6,433 tokens/second**. The GPU was idle between
the final runs. This is about 12.8 times the historical native run below, although
the optimizer, physical batch and precision differ, so it is not a controlled
kernel benchmark. The run completed the original OneCycle schedule with 100
planned updates.

All evaluations used the same 128 random held-out contexts as the independent
Transformers comparison: 65,536 all-position targets and 128 final-position
targets representing 477 bytes.

| Validation metric | Original model | After 100 updates |
| --- | ---: | ---: |
| All-token loss | 3.10103 | 2.74160 |
| BPB | 1.26735 | 1.12046 |
| Last-token loss | 3.06295 | 2.94395 |
| Last bpb | 1.185786 | 1.139715 |

All-token validation loss improved 11.6%; last bpb improved 3.89%. Restoring the
final checkpoint reproduced every validation metric exactly to the logged
precision. `runs/native-muon-tf32x2-hn1g/` contains the checkpoint, logs, TensorBoard
events, plots, exact executable, source snapshot and SHA-256 manifest; restoration evidence
is in `runs/native-muon-tf32x2-hn1g-restore/`. The independent comparison reports and
shared windows are under `runs/transformers-parity-final/`.

An earlier FP32-fallback run under concurrent GPU load remains in
`runs/native-muon-hn1g/`: 113 updates, 3,405 tokens/second, final loss 2.74112 and
last bpb 1.130951. It used a 200-update OneCycle horizon and stopped at its
600-second limit. Its arithmetic, schedule and GPU load differ from the final run.

## Historical native run

On the RTX PRO 6000, 142 updates processed 290,816 training tokens in about
10.2 minutes including validation, at approximately 503 training tokens/second.
The run used context 512, accumulation 4, rank 16, and 32 fixed validation blocks
(16,384 target tokens). It was recovered after a PC restart by replaying the
first 50 updates to replace incomplete optimizer state, then resuming from a
checksummed, durably saved checkpoint.

| Validation metric | Initial | After 142 updates |
| --- | ---: | ---: |
| All-token loss | 3.15250 | 2.80129 |
| BPB | 1.30382 | 1.15857 |
| Last-token loss | 3.06598 | 2.96497 |
| Last-token BPB | 1.14149 | 1.10388 |

All-token validation loss improved 11.1%. Last-token loss improved more modestly
and fluctuated between evaluations. Peak arena usage stayed at 16.59 GiB.
The final checkpoint was reloaded and reproduced all validation metrics exactly
to the logged precision. Logs, TensorBoard events, checkpoint, CSV and plots are
saved locally under `runs/native-hn1g/`; `summary.json` combines both clean run
segments. The superseded interrupted attempt is under
`runs/native-hn1g-interrupted/`.

## Historical corrected-TF32 batch and PyTorch comparison

The physical-batch sweep holds the optimizer batch at **128 sequences / 65,536
predicted tokens**, context 512, rank 16, alpha 32 and all 186 projection targets.
Both programs consume identical hn1g windows and identical initial adapters.
Six Muon updates run per measurement; the reported throughput excludes the first
two updates, loading, evaluation and checkpoint writes. Strata was stopped for
these measurements. The RTX retained its configured **330 W power limit**;
the driver reported active software power limiting during training.

| Backend / arithmetic | Physical batch | Accumulation | Tokens/sec | Peak storage used, GiB |
| --- | ---: | ---: | ---: | ---: |
| Native corrected TF32 / FP32 | 16 | 8 | 6,078 | 12.53 |
| Native corrected TF32 / FP32 | 32 | 4 | 5,972 | 20.89 |
| Native corrected TF32 / FP32 | 64 | 2 | 5,873 | 37.73 |
| Native corrected TF32 / FP32 | 128 | 1 | 5,943 | 71.42 |
| Native IEEE FP32 | 32 | 4 | 5,283 | 20.89 |
| Native plain TF32 | 32 | 4 | 9,033 | 20.89 |
| PyTorch IEEE FP32, reference DeltaNet, SDPA | 32 | 4 | 4,124 | 12.26 |
| PyTorch BF16, FLA, SDPA | 32 | 4 | 12,399 | 10.04 |
| PyTorch BF16, FLA, SDPA | 128 | 1 | 11,728 | 12.64 |

Memory is native arena storage versus PyTorch allocator storage, with CUDA
driver/context memory additional in both cases. These are short throughput
measurements; small differences between physical batches do not establish a
lasting advantage. Increasing the physical batch did not improve speed on either
path. Native IEEE FP32 was 28% faster than PyTorch's IEEE reference, while the
optimized PyTorch BF16 path was 2.08 times the then-default native throughput.
The BF16 row uses different precision and different DeltaNet/attention kernels;
it is not an inference-equivalence claim or a convergence comparison.

The corrected-TF32 native mode retains additional input precision through two frozen
projection products and FP32 attention/adapters. Plain TF32 removes that work
and increases throughput about 51%, with the accuracy tradeoff documented above.
The full-vocabulary head always processes at most 2,048 rows per chunk, so larger
physical batches still execute the same total head chunks per optimizer update.
Layer recomputation and recurrent-state traffic also remain. The larger-batch
hardware-counter capture measured **34.99% average tensor-pipeline activity and
45.36% average DRAM throughput** over the full batch-128 update; it does not
establish whole-update peak saturation. Report and counters are in
`runs/profile-fixed-batch128-update.ncu-repz` and
`runs/batch-comparison/hardware-counters.json`.

`runs/batch-comparison/` contains the shared windows and SHA-256, per-update
durations/losses, benchmark reports, CSV/JSON comparison, PNG/SVG plot, source
snapshot and executable. A native cuBLAS `FAST_16BF` experiment measured 9,089
tokens/sec with a 1 GiB workspace and 8,969 with 2 GiB. It gave no useful gain
over plain TF32 and was removed; it used FP32 input/storage with library downconversion,
unlike the current packed BF16 path. Its source, executable,
measurements and kernel timeline are retained under `experiments/bf16-gemm/`.

The comparison environment uses installed PyTorch/Transformers and optional
[FLA](https://github.com/fla-org/flash-linear-attention) /
[Triton for Windows](https://github.com/triton-lang/triton-windows). It does not
add neural-framework dependencies to the native trainer:

```powershell
uv pip install --python validation/.venv/Scripts/python.exe -r validation/requirements-benchmark.txt
# Choose a fresh output directory; all invocations share the same windows.
uv run --locked python validation/benchmark_training.py --backend native --batch 32 --output runs/batch-comparison-new
uv run --locked python validation/benchmark_training.py --backend native --batch 128 --vram-gib 80 --output runs/batch-comparison-new
validation/.venv/Scripts/python.exe validation/benchmark_training.py --backend torch --precision fp32 --batch 32 --output runs/batch-comparison-new
validation/.venv/Scripts/python.exe validation/benchmark_training.py --backend torch --precision bf16 --delta fla --batch 32 --output runs/batch-comparison-new
uv run --with matplotlib python tools/summarize_benchmarks.py runs/batch-comparison-new
```

PyTorch takes its identical initial adapters from the native optimizer probe
export at `runs/optimizer-parity-final/muon/initial.f32`. On a fresh checkout,
run the native executable with `--optimizer-probe PATH`, then pass that directory's
`initial.f32` through the benchmark's `--initial-adapter` option. Its reference
head accumulates exact chunk gradients before backpropagating through the
backbone once, avoiding a complete `[batch*context,248320]` vocabulary tensor.

## Metrics and checkpoints

Each update logs JSON to stdout and `metrics.jsonl`; evaluation uses identical
blocks each time. TensorBoard events use the original C++ logger and include
`train/` and `eval/` tags for loss, BPB, last-token loss, last-token BPB, learning
rate, gradient norm and peak arena usage. `report.json` records final metrics,
speed and configuration. `assets.json` preserves model/data provenance.

* **Loss:** mean negative log probability per predicted BPE token, in nats.
* **BPB:** summed negative log probability divided by `ln(2)` and the UTF-8 byte
  lengths of the tokenizer-normalized target tokens. Partial UTF-8 tokens are
  counted correctly from the byte-level vocabulary, without decoding each token.
* **Last-token loss:** loss at the final prediction in each context, averaged
  across contexts. Validation uses 16 final predictions by default, or 32 in the
  example command. This is noisier than all-token loss.

These metrics use Qwen BPE tokenization and a different context length from the
original byte-level experiment; interpret direct comparisons accordingly.
Across the complete corpus, NFC normalization reduces the training byte count
by two bytes out of 990,000,029; validation byte accounting matches all 9,999,971
source bytes.

`checkpoint.bin` saves adapters, optimizer state, completed update count and
sampling RNG state. Base weights remain in the Hugging Face cache. A checkpoint
is saved at each evaluation and at completion, with a checksum, durable storage
flush and atomic replacement. An incomplete or altered checkpoint is rejected.
Resume with matching context, accumulated total batch, optimizer, rank and
alpha; physical batch and accumulation may change together. `--steps` is the total desired update count. Keep the original planned
step count for an exact OneCycle continuation; changing it changes the schedule.
For the historical AdamW run:

```powershell
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--batch','1','--accum','4','--optimizer','adamw','--precision','tf32','--steps','1000','--seconds','600','--resume','runs/native-hn1g/checkpoint.bin','--output','runs/native-hn1g')
```

The checkpoint is a native trainer format, not a PEFT or Hugging Face export.
Format version 3 records optimizer and physical batch; version 2 is read with
legacy AdamW/batch-1 configuration. Pre-checksum version 1 files are rejected.
`report.json` describes the current invocation; `tools/plot_run.py`
combines JSON logs across resumptions and produces `summary.json`, a validation
CSV and PNG/SVG loss plots:

```powershell
uv run --with matplotlib python tools/plot_run.py runs/native-hn1g
```

## Verification

```powershell
uv run --locked pytest -q
./build.ps1 -CudaArch 120 -Run -TrainingArgs @('--device','1','--self-test')
```

The GPU self-test checks recurrent DeltaNet derivatives, composed normalization,
convolution, rotary embeddings and residuals, grouped causal attention, query
gating, and both input and adapter LoRA gradients against finite differences.
Asset tests check chronological record boundaries, UTF-8 preservation, cache
invalidation and exact byte accounting for BPB. The historical Rust oracle tests
in `tests/reference.py` are outside this native test suite.
BF16 tests independently round operands on the CPU and check batched/transposed
GEMMs, accumulation, scratch reclamation, cached LoRA forward/input/adapter
gradients and fused cross-entropy gradients. The scalar BF16 LoRA oracle matches
exactly; GEMM maximum error is 1.43e-6.

## Independent pip reference validation

Install the reference packages in a separate environment. These packages are
never loaded by the trainer:

```powershell
uv venv validation/.venv --python 3.12
uv pip install --python validation/.venv/Scripts/python.exe torch==2.14.1 --index-url https://download.pytorch.org/whl/cu132
uv pip install --python validation/.venv/Scripts/python.exe -r validation/requirements.txt
validation/.venv/Scripts/python.exe validation/compare.py --samples 128 --context 512 --output runs/transformers-parity-final
validation/.venv/Scripts/python.exe validation/compare.py --samples 128 --context 512 --precision tf32x2 --output runs/transformers-parity-tf32x2-final
validation/.venv/Scripts/python.exe validation/check_optimizers.py
uv run --locked python validation/check_native_training.py
```

`compare.py` loads the installed Transformers `Qwen3_5ForCausalLM` directly,
using its built-in eager/reference implementations. It checks final hidden
states at every position and logits, all 186 zero adapters, top predictions,
and original-model last bpb. Add `--all-logits --context 16 --samples 4` for
comparison of every vocabulary logit at every position. The default compares
native IEEE FP32 with IEEE FP32 Transformers; `--precision tf32x2` checks
the corrected reference arithmetic against the same reference and strict tolerance.
Plain TF32 has a separate tolerance for its lower input precision and is retained
as an optional throughput experiment.

The validator draws token offsets uniformly without replacement from the
chronological hn1g holdout, using seed 20261009. `inputs.bin` contains the exact
same input/target windows for both programs. `inputs.json` records offsets,
dataset/model provenance and SHA-256. Native CUDA computes each last-token NLL
and C++ aggregates last bpb; the Python reference computes NLL independently
with float64 log-sum-exp. It additionally checks the CUDA NLL against the
exported native logits. Pass `--eval-input runs/transformers-parity-final/inputs.bin`
to train with repeated evaluation on this identical random subset.

Last bpb is **sum of final-position NLL / (ln(2) * sum of final-target bytes)**,
not the mean of per-token BPB ratios. Every context contributes one final
prediction. All-token BPB still includes all predicted positions.

The verified 128-context FP32 result is **1.185787281 last bpb in Transformers**
versus **1.185786322 in CUDA/C++**, an absolute difference of **9.6e-7**. All 128
top predictions match. Maximum logit error is 9.11e-5 and hidden-state error
1.76e-4; native zero adapters preserve every output bit. Python zero adapters
also preserve the checked outputs exactly. A separate all-position comparison
checks every logit on four 16-token contexts. The optional BF16x9 setting passes
the strict FP32 comparison through its native FP32 fallback on this GPU.

With corrected TF32 arithmetic, the same subset produces
**1.185785981 last bpb in CUDA/C++**, differing from the Python reference by
**1.3e-6**. All 128 top predictions still match. Maximum hidden error is 0.00123
and maximum logit error is 0.000265; zero adapters remain bit-identical.
`runs/transformers-parity-tf32x2-final/comparison.json` contains the report;
`runs/transformers-parity-tf32x2-all-logits/` checks every logit on short contexts.

The current build repeats that strict comparison in
`runs/transformers-parity-strict-bf16-build/`: the corrected/FP32 reference path
retains the same 1.3e-6 last-bpb difference and exact zero-adapter identity.
For a separate measurement of BF16 rounding against the original IEEE model:

```powershell
validation/.venv/Scripts/python.exe validation/compare.py --precision bf16 --accuracy-only --samples 128 --context 512 --output runs/bf16-accuracy-new
```

On the identical subset, native BF16 last bpb is **1.185840072**, versus
**1.185787281** in IEEE Transformers: absolute difference **0.00005279**.
All 128 top predictions match, and zero adapters preserve every native hidden
state/logit bit. BF16 logit RMSE is 0.01053, maximum logit error 0.131 and maximum
row-scaled hidden error 3.175%. That outlier exceeds the preliminary 3% bound;
the retained report is an accuracy measurement, not strict IEEE parity.
`--accuracy-only` still asserts zero-adapter identity and CUDA NLL correctness;
it records numerical bounds without asserting cross-precision equivalence.
`--reference-dtype bf16` measures against pip Transformers' full-BF16 preset;
`bf16-gemm` instead wraps pip linear/attention products with BF16 operands and
FP32 output while leaving its model/state operations intact. BF16 rounding can
amplify tiny operation-order differences even with the latter policy.

Training checks cover full versus recomputed activations, physical batching
versus gradient accumulation, captured versus uncaptured execution, cached graph
replay, MantissaAdamW versus ordinary AdamW and exact Muon checkpoint restoration.
FP32 checks use tighter error bounds. Corrected TF32 changes reductions with GEMM
geometry: after three updates, batching differs by 0.0252% of momentum L2 norm
and 1.59e-5 validation loss. Replay/recomputation and optimizer-representation
checks retain the stricter bounds. Detailed error statistics are saved under
`runs/native-training-checks-tf32x2-final/`.
BF16 multi-update training checks use separate parameter/state/model-loss bounds:
small reduction-order or clipping differences can cross BF16 rounding boundaries
on subsequent updates. They do not claim bit-identical training across batch,
recomputation or capture choices. Optimizer-equation parity remains independently
tested with identical supplied gradients.

## Original optimizers

`native/optim.h` is copied unchanged from the original TurboGPT `optim.h`.
`--schedule onecycle` uses its cosine LR and momentum schedule. The default
`warmup` schedule keeps the peak LR after three updates for time-limited runs.

`native/optimizers.cuh` reuses `precise_weight`, `write_weight`, AdamW equations,
Muon EMA/Nesterov momentum, BF16 normalization, five Newton-Schulz iterations
with coefficients (3.4445, -4.775, 2.0315), matrix orientation and LR shape
scaling from TurboGPT's `cuda/step.cuh`. Shape-grouped cuBLAS replaces its
fixed-width WMMA layout to support rank-16 adapters. The original MIT notice is
in `native/LICENSE-turboGPT`.

The original bit representation splits an FP32 weight into BF16 high bits and
16 low mantissa bits, preserving every weight bit. Its Adam momentum/variance
are FP32; it is not an 8-bit-moment optimizer. Native projections use an exact
FP32 view of the split weight. Muon uses only momentum; its second-state slot
is retained for checkpoint compatibility. Across 20 updates on all 372 adapter
matrices, native Muon differs from pip `torch.optim.Muon` by 0.041% of update
L2 norm; MantissaAdamW differs from pip `torch.optim.AdamW` by 0.000021%.

## Hardware-counter profiling

Build first, then use the installed NVIDIA profilers:

```powershell
./tools/profile_native.ps1 -Mode Timeline -Output runs/profile-native-timeline
./tools/profile_native.ps1 -Mode Delta -Output runs/profile-native-delta
./tools/profile_native.ps1 -Mode Gemm -Output runs/profile-native-gemm
./tools/profile_native.ps1 -Mode Range -Output runs/profile-native-update
uv run python tools/summarize_profiles.py runs/profile-native-delta.csv runs/profile-native-gemm.csv runs/profile-native-update.csv
```

The range profile measures the complete third update using application range
replay, including recomputation, backward, clipping and Muon. Kernel profiles
identify which resource limits each kernel. Reports are `.ncu-repz` plus raw
counter CSV, or `.nsys-rep` for the timeline. `--profile-step N` brackets a
specific update using CUDA's driver profiler API. Application range replay
supports cuBLAS's CUDA 13 library attribute calls that ordinary range capture
cannot replay. The profiler intentionally keeps the existing GPU clocks.
Timeline capture enables graph-node tracing so kernels inside the cached graph
are visible. JSON sidecars record executable hashes, commands and the GPU load,
clocks and power limit before capture.

Compare counter-derived SM, tensor-pipeline and DRAM throughput against their
own sustained peaks and inspect occupancy/eligible warps, rather than treating
`nvidia-smi`'s busy percentage as compute utilization. Counter interpretation
and replay behavior are described in NVIDIA's
[profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

The original DeltaNet launch had 16 blocks and 1.26% SM throughput. Tiling,
layer recomputation, short register replay and derivative reduction support
physical batch 32 within the 22 GiB arena. A measured register-replay backward
kernel reaches 75.31% DRAM throughput, versus the original forward's 0.09%.
This is a kernel measurement; whole-update throughput is reported separately.
The RTX PRO 6000 is also used by other local workloads, so elapsed time and
whole-update counters under concurrent use do not establish isolated peak
performance. These historical captures retained concurrent workloads. For the
subsequent controlled comparison above, Strata was stopped with the user's authorization.

For the final corrected-TF32 configuration, a large projection GEMM reaches
**87.83% tensor-pipeline throughput**, while the DeltaNet backward kernel reaches
**80.02% DRAM throughput** (8.49 ms, 40.12% occupancy). These are measurements of
different bottlenecks, not percentages to add together. The reports are
`runs/profile-tf32x2-gemm.ncu-repz` and `runs/profile-tf32x2-delta.ncu-repz`;
the full update is profiled separately in `runs/profile-tf32x2-update.ncu-repz`.
The final range began with GPU utilization at 0% and measures **38.82% average
tensor throughput and 44.86% average DRAM throughput over 2.553 seconds**.
Those averages include phases with different resource limits; the individual
kernel results do not establish whole-step peak saturation.
An additional timeline captured after the GPU became idle records **2.422 seconds
of kernel execution in a 2.467-second kernel span (98.14%)**, showing small gaps
between kernels. Its report is `runs/profile-tf32x2-idle-timeline.nsys-rep`.
The existing 330 W power limit is recorded and left unchanged.

For the current BF16 build, the full batch-16/accumulation-8 update measures
**19.66% tensor-pipeline activity, 52.73% DRAM throughput and 30.40% SM throughput**
over a profiled 5.632 seconds. These averages show remaining phase-specific
bottlenecks and do not claim peak GPU saturation. Range collection required
seven application replays; the profiled duration includes instrumentation effects.
The separate timeline has 47,290 kernels, 5.288 seconds of summed execution
in a 5.463-second kernel span (**96.80% activity within that span**). DeltaNet
forward/backward together account for 30.1% of kernel time. Counter percentages
and kernel activity have different meanings; throughput comparisons above use
unprofiled updates. Captures and summaries are under `runs/bf16-performance-final/`.
Its large BF16 projection reaches **76.67% tensor-pipeline throughput**
(284 microseconds); DeltaNet backward reaches **54.03% DRAM throughput**
(7.245 milliseconds, 29.75% occupancy). These are individual-kernel scopes.
