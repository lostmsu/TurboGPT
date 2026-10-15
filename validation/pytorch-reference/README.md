# Qwen-Tuning

Archived PyTorch comparison implementation. Current training uses the native
CUDA C++ code in the repository root. This directory preserves the earlier
implementation, dependency lockfile, and tests for porting and comparison.
Run its commands from this directory using its separate Python environment.

Continue pretraining **Qwen/Qwen3.5-0.8B-Base** on the raw **hn1g** Hacker News
text stream using BF16 LoRA or NF4 QLoRA. Defaults target one 24 GiB NVIDIA GPU
with a **22 GiB PyTorch allocator budget**, leaving room for the driver.

The original byte-level CUDA Rust experiment remains in `src/`; its kernels and
checkpoints cannot represent Qwen's hybrid architecture. The Qwen path uses
Transformers/PEFT. See [README-turbogpt.md](README-turbogpt.md) for the Rust tools.

## Setup

Use Python 3.12/3.13, [uv](https://docs.astral.sh/uv/), and an NVIDIA driver
supporting CUDA 13.2. The lockfile selects CUDA PyTorch on Windows and Linux.
Native Windows training works independently of the Rust WSL toolchain.

```powershell
uv sync --locked
uv run --locked qwen-tune download
uv run --locked qwen-tune prepare
uv run --locked qwen-tune train --offline --device 0
```

The full model snapshot is pinned to
`dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68` in the standard Hugging Face cache
layout at **`.cache/huggingface/hub/`**. Override with `--cache-dir`.
`--offline` requires cached model/dataset files; omit it to permit downloads.
The loader extracts the text backbone, omits vision, and rejects missing or
mismatched pretrained text weights. No remote Python model code is used.

`--device` selects one GPU using PCI bus order (`nvidia-smi`). On this development
machine, **use `--device 1`** for the RTX PRO 6000; GPU 0 is a busy 16 GiB card.
Training uses that one GPU, with no CPU offload.

## Dataset

Data discovery checks `./hn1g.txt`, then `~/Downloads/Datasets/hn1g.txt`, then
downloads only `hn1g.txt` from
[lostmsu/hn1g](https://huggingface.co/datasets/lostmsu/hn1g). Use `--data PATH` for
another local text file. Published hn1g is 1,000,000,000 bytes, SHA-256:
`000beb89de92941829a2f64ee3f0f500de92c98959dd47e97b9a7347b9b285b9`.

This trains next-token prediction on the original text, whitespace and markup;
it does not create chat conversations. The final 1% of bytes is a chronological
holdout, with the split moved to the next record header. Train and validation
are tokenized independently. Multiline bodies and older files without newlines
between records are supported. Invalid UTF-8, if present, is replaced on decode.

Bounded tokenization chunks produce uint32 files in `.cache/tokens/`. Training
memory-maps these files. Cache keys include source hash, model revision, tokenizer
and split settings; a manifest records boundaries and token counts. Adjacent
blocks share one context token, training each next-token pair exactly once.
The final incomplete block is dropped. Validation uses up to 64 full blocks from
the start of the holdout, without storing vocabulary logits.

The verified local corpus is fully prepared: **281,186,047 train tokens** and
**2,838,182 validation tokens**, with the byte split at 990,000,029.

## 24 GiB configuration

| Setting | Default |
| --- | --- |
| Base | Frozen BF16 text weights, tied embeddings/output head |
| Adapter | LoRA rank 16, alpha 32, no dropout |
| Targets | All text linear layers, including Gated DeltaNet and MLP |
| Context / microbatch | 2,048 tokens / 16 sequences |
| Accumulation | 1 microbatch: 32,752 predicted tokens per update |
| Loss | Exact causal cross-entropy in 128-token vocabulary chunks |
| Activations | Non-reentrant gradient checkpointing, no KV cache |
| Optimizer | Adapter-only AdamW, LR 1e-4, cosine decay, 3% warmup |
| Duration | One epoch; `--max-steps N` overrides |
| Evaluate / checkpoint | Every 250 steps and at completion; retain two checkpoints |
| PyTorch allocator budget | 22 GiB |

BF16 LoRA is the default because the frozen 0.8B weights fit comfortably without
quantization overhead. Qwen's large vocabulary is a more significant activation
cost. The trainer checkpoints both the output projection and cross-entropy so
autograd does not retain all chunks' vocabulary logits until backward. Tests
verify the same loss and gradients as ordinary cross-entropy.

The memory budget caps PyTorch's reserved memory. CUDA contexts, driver and
external kernel allocations are outside it; other processes also consume VRAM.
A 24 GiB card has roughly 2 GiB outside this budget. Runs report measured peak
allocated and reserved VRAM. Re-measure larger contexts/microbatches; reduce
`--sequence-length` or `--batch-size` if memory is exhausted, and increase
`--gradient-accumulation` to preserve the effective batch size.

The default BF16 configuration was tested on the RTX PRO 6000 Blackwell under
the 22 GiB cap: two updates at batch 16/context 2,048 peaked at **12.29 GiB
reserved / 11.81 GiB allocated**. It took 18.67 seconds versus 40.56 seconds for
batch 1/accumulation 16 with the same effective batch. These short runs include
startup overhead and do not measure a full epoch. For a lower-memory fallback,
use `--batch-size 1 --gradient-accumulation 16` (2.61 GiB peak reserved).

For 4-bit NF4 QLoRA with double quantization and BF16 compute:

```powershell
uv sync --locked --extra qlora
uv run --locked --extra qlora qwen-tune train --offline --device 0 --method qlora --output-dir runs/qwen-hn1g-qlora
```

The same batch-16 GPU probe used **15.51 GiB reserved** with QLoRA versus
**12.29 GiB** with BF16 LoRA. Quantization preparation upcasts nonquantized
weights and introduces different intermediate buffers, so NF4 does not reduce
the total training peak for this small model in this configuration.

Transformers' reference Gated DeltaNet kernels work without optional packages
but are slower. On supported Linux systems, the official
[Qwen3.5 docs](https://huggingface.co/docs/transformers/model_doc/qwen3_5) describe
`flash-linear-attention` and `causal_conv1d` for faster kernels. They are optional;
the default Windows configuration uses the correct PyTorch fallbacks.

## Outputs and resume

`runs/qwen-hn1g/` contains `run_config.json` (settings, source hash, split,
revision, GPU and versions), `checkpoint-N/` (adapter, tokenizer, optimizer,
scheduler, RNG and Trainer state), `adapter/` (final adapter/tokenizer), and
`all_results.json`/`trainer_state.json` (loss, perplexity, speed and VRAM).

```powershell
uv run --locked qwen-tune train --offline --device 0 --resume runs/qwen-hn1g/checkpoint-250
```

Resume with the same dataset, split, revision, adapter and batch configuration;
the CLI checks them against the checkpoint's run. Existing output directories
require `--resume`; new experiments should use a new `--output-dir`.
Loss is nats per Qwen token and is not directly comparable to the Rust model's
bits per byte. This setup does not enable multi-GPU training.

For inference, load the text backbone explicitly (the original Hub config also
includes vision), then attach the adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoConfig, Qwen3_5ForCausalLM
from qwen_tuning import DEFAULT_CACHE, MODEL_ID, MODEL_REVISION

cached = dict(revision=MODEL_REVISION, cache_dir=DEFAULT_CACHE, local_files_only=True)
base = Qwen3_5ForCausalLM.from_pretrained(
    MODEL_ID, config=AutoConfig.from_pretrained(MODEL_ID, **cached).text_config,
    key_mapping={r"^model\.language_model\.": "model."},
    dtype=torch.bfloat16, device_map={"": 0}, **cached,
)
model = PeftModel.from_pretrained(base, "runs/qwen-hn1g/adapter").eval()
```

Choose the appropriate GPU in `device_map`. The adapter records its base
ID/revision. The custom loss does not change generation logits.

## Verification

```powershell
uv run --locked pytest -q
uv run --locked qwen-tune train --offline --device 1 --max-data-bytes 4194304 --validation-fraction 0.1 --max-steps 2 --eval-blocks 2 --eval-steps 2 --save-steps 1 --logging-steps 1 --output-dir runs/smoke
```

The smoke run uses the default context/rank, performs updates, evaluates and
saves checkpoints. Its small token cache is separate from the full corpus cache.
Repeat with `--max-steps 4 --resume runs/smoke/checkpoint-2` to check resuming.
