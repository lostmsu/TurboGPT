"""Controlled native / pip Transformers LoRA throughput comparison.

This is a reference benchmark, never a production trainer. Both paths consume
the same windows, rank/targets, effective batch and exact full-vocabulary loss.
FP32 and BF16 results are reported separately. No model code is copied here.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import struct
import subprocess
import sys
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["USE_HUB_KERNELS"] = "NO"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from qwen_tuning import DEFAULT_CACHE, MODEL_REVISION


def gpu_state(device):
    return subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,clocks.sm,power.limit",
        "--format=csv,noheader", "-i", str(device)], text=True).strip()


def shared_inputs(args):
    directory = args.output / "shared"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "inputs.bin"
    assets = json.loads((ROOT / ".cache/native/assets.json").read_text())
    samples = args.steps * args.total_batch
    corpus = np.memmap(assets["paths"]["train"], dtype="<u4", mode="r")
    starts = np.random.default_rng(args.seed).choice(len(corpus) - args.context, samples, replace=False)
    ids = np.stack([corpus[s:s + args.context + 1] for s in starts]).astype("<u4")
    content = struct.pack("<QII", 0x514e494e50555431, samples, args.context) + ids.tobytes()
    if path.exists() and path.read_bytes() != content:
        raise ValueError("Existing shared inputs differ; select a new output directory")
    path.write_bytes(content)
    info = {"sha256": hashlib.sha256(content).hexdigest(), "seed": args.seed, "context": args.context,
            "samples": samples, "total_batch": args.total_batch, "start_token_offsets": starts.tolist(),
            "model_revision": MODEL_REVISION, "dataset": assets["dataset"]}
    (directory / "inputs.json").write_text(json.dumps(info, indent=2) + "\n")
    return path, ids, assets, info


def native(args, path, info):
    suffix = f"-w{args.blas_workspace_mib}" if args.blas_workspace_mib != 1024 else ""
    directory = args.output / f"native-{args.precision}-b{args.batch}{suffix}"
    cuda = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"))
    env = os.environ.copy()
    env["PATH"] = str(cuda / "bin") + os.pathsep + str(cuda / "bin/x64") + os.pathsep + env["PATH"]
    command = [str(ROOT / "build/qwen-tune.exe"), "--device", str(args.device), "--context", str(args.context),
               "--batch", str(args.batch), "--steps", str(args.steps),
               "--train-input", str(path), "--precision", args.precision, "--optimizer", "muon", "--eval-samples", "1",
               "--eval-every", str(args.steps + 1), "--vram-gib", str(args.vram_gib), "--output", str(directory),
               "--benchmark-warmup", str(args.warmup), "--blas-workspace-mib", str(args.blas_workspace_mib)]
    before = gpu_state(args.device)
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    report = json.loads((directory / "report.json").read_text())
    rows = [json.loads(row) for row in (directory / "metrics.jsonl").read_text().splitlines()]
    updates = [row for row in rows if row["kind"] == "train"]
    report.update({"backend": "native", "input_sha256": info["sha256"], "gpu_before": before,
                   "gpu_after": gpu_state(args.device), "command": command,
                   "executable_sha256": hashlib.sha256((ROOT / "build/qwen-tune.exe").read_bytes()).hexdigest(),
                   "update_seconds": [row["update_seconds"] for row in updates],
                   "training_losses": [row["loss"] for row in updates],
                   "median_steady_update_seconds": statistics.median(row["update_seconds"] for row in updates[args.warmup:])})
    (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def pytorch(args, ids, assets, info):
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoConfig, Qwen3_5ForCausalLM

    directory = args.output / f"torch-{args.precision}-{args.delta}-b{args.batch}"
    directory.mkdir(parents=True, exist_ok=False)
    before = gpu_state(args.device)
    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    torch.manual_seed(3407)
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high" if args.precision == "tf32" else "highest")
    torch.backends.cuda.matmul.allow_tf32 = args.precision == "tf32"
    torch.backends.cudnn.allow_tf32 = args.precision == "tf32"
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    from transformers.models.qwen3_5 import modeling_qwen3_5
    if args.delta == "reference":
        # Transformers automatically selects an installed FLA package, even if
        # Hub kernels are disabled. Explicitly select its pip reference for FP32.
        modeling_qwen3_5.torch_chunk_gated_delta_rule = inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)
    if args.delta == "fla":
        if dtype != torch.bfloat16:
            raise ValueError("FLA's low-precision tensor kernels are benchmarked only with --precision bf16")
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # Select the installed FLA function explicitly, without downloading Hub kernels.
        modeling_qwen3_5.torch_chunk_gated_delta_rule = chunk_gated_delta_rule
    snapshot = DEFAULT_CACHE / "models--Qwen--Qwen3.5-0.8B-Base" / "snapshots" / MODEL_REVISION
    config = AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config
    config.use_cache = False
    model, loading = Qwen3_5ForCausalLM.from_pretrained(
        snapshot, config=config, dtype=dtype, device_map={"":args.device}, attn_implementation="sdpa",
        local_files_only=True, output_loading_info=True, key_mapping={r"^model\.language_model\.":"model."})
    assert not loading.get("missing_keys") and not loading.get("mismatched_keys"), loading
    model.requires_grad_(False)

    class LoRA(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
            self.a = torch.nn.Parameter(torch.empty(16, base.in_features, device=device, dtype=torch.float32))
            self.b = torch.nn.Parameter(torch.zeros(base.out_features, 16, device=device, dtype=torch.float32))
            torch.nn.init.uniform_(self.a, -base.in_features ** -.5, base.in_features ** -.5)

        def forward(self, x):
            return self.base(x) + 2 * F.linear(F.linear(x, self.a.to(x.dtype)), self.b.to(x.dtype))

    projections = sorted((name, layer) for name, layer in model.named_modules()
                         if name.startswith("model.layers.") and isinstance(layer, torch.nn.Linear))
    assert len(projections) == 186
    # The native optimizer probe exports initialization in std::map projection order.
    # Reusing it makes zero-B adapter initialization identical across frameworks.
    initial_path = args.initial_adapter
    if initial_path is None:
        raise ValueError("Supply --initial-adapter from a native --optimizer-probe export")
    initial = np.fromfile(initial_path, dtype="<f4")
    offset = 0
    adapters = []
    for name, layer in projections:
        lora = LoRA(layer)
        parent, leaf = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), leaf, lora)
        for p in (lora.a, lora.b):
            count = p.numel()
            with torch.no_grad():
                p.copy_(torch.as_tensor(initial[offset:offset+count].reshape(p.shape), device=device))
            offset += count
            adapters.append(p)
    assert offset == len(initial) == 10_822_656
    optimizer = torch.optim.Muon(adapters, lr=.001, momentum=.95, weight_decay=.01, ns_steps=5)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.train()
    lengths = np.fromfile(assets["paths"]["token_bytes"], dtype="<u4")
    losses, last_losses, bpbs, last_bpbs, times = [], [], [], [], []
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    accumulation = args.total_batch // args.batch
    for step in range(args.steps):
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        parts, final_parts = [], []
        for micro in range(accumulation):
            begin = step * args.total_batch + micro * args.batch
            windows = ids[begin:begin+args.batch].astype(np.int64)
            input_ids = torch.as_tensor(windows[:, :-1].copy(), device=device)
            targets = torch.as_tensor(windows[:, 1:].copy().reshape(-1), device=device)
            hidden = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
            # The frozen head has no trainable weights. Compute its exact gradient
            # chunk by chunk, then backpropagate through the backbone once. This
            # avoids retaining the complete [B*T,248320] vocabulary tensor, just
            # as the native trainer does; no additional head recomputation.
            flat = hidden.detach().reshape(-1, hidden.shape[-1])
            hidden_grad = torch.empty_like(flat)
            for row in range(0, len(targets), args.head_chunk):
                states = flat[row:row+args.head_chunk].detach().requires_grad_(True)
                logits = model.lm_head(states).float()
                nll = F.cross_entropy(logits, targets[row:row+args.head_chunk], reduction="none")
                part = nll.sum() / (args.total_batch * args.context)
                hidden_grad[row:row+args.head_chunk] = torch.autograd.grad(part, states)[0]
                parts.append(part.detach())
                last = torch.arange(row, row+len(nll), device=device) % args.context == args.context-1
                final_parts.append(nll.detach()[last].sum())
                del logits, nll, states
            hidden.backward(hidden_grad.reshape_as(hidden))
            del hidden, hidden_grad, flat
        norm = torch.nn.utils.clip_grad_norm_(adapters, 1.0)
        for group in optimizer.param_groups:
            group["lr"] = .001 * min(1, (step + 1) / 3)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        loss = torch.stack(parts).sum().item()
        last_loss = torch.stack(final_parts).sum().item() / args.total_batch
        if not np.isfinite(loss) or not torch.isfinite(norm):
            raise RuntimeError("Non-finite reference training loss or gradient")
        windows = ids[step*args.total_batch:(step+1)*args.total_batch]
        bpb = loss*args.total_batch*args.context/(math.log(2)*int(lengths[windows[:,1:]].sum(dtype=np.uint64)))
        last_bpb = last_loss*args.total_batch/(math.log(2)*int(lengths[windows[:,-1]].sum(dtype=np.uint64)))
        times.append(elapsed); losses.append(loss); last_losses.append(last_loss); bpbs.append(bpb); last_bpbs.append(last_bpb)
        print(json.dumps({"backend":"torch", "step":step+1, "update_seconds":elapsed,
                          "loss":loss, "bpb":bpb, "last_loss":last_loss, "last_bpb":last_bpb, "grad_norm":norm.item(),
                          "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30}), flush=True)
    steady_seconds = sum(times[args.warmup:])
    report = {"backend":"torch", "torch":torch.__version__, "transformers":transformers.__version__,
              "precision":args.precision, "delta_kernel":args.delta, "attention":"SDPA", "checkpointing":"layer",
              "batch":args.batch, "total_batch":args.total_batch, "accumulation":accumulation, "context":args.context,
              "rank":16, "alpha":32, "trainable_parameters":offset, "optimizer":"pip torch.optim.Muon",
              "head_chunk":args.head_chunk, "steps":args.steps, "benchmark_warmup_updates":args.warmup,
              "training_seconds":sum(times), "update_seconds":times, "training_losses":losses, "training_last_losses":last_losses,
              "training_bpbs":bpbs, "training_last_bpbs":last_bpbs,
              "steady_training_seconds":steady_seconds, "steady_training_tokens":(args.steps-args.warmup)*args.total_batch*args.context,
              "steady_tokens_per_second":(args.steps-args.warmup)*args.total_batch*args.context/steady_seconds,
              "median_steady_update_seconds":statistics.median(times[args.warmup:]),
              "peak_allocated_gib":torch.cuda.max_memory_allocated(device)/2**30,
              "peak_reserved_gib":torch.cuda.max_memory_reserved(device)/2**30,
              "input_sha256":info["sha256"], "initial_adapter_sha256":hashlib.sha256(initial_path.read_bytes()).hexdigest(),
              "gpu_before":before, "gpu_after":gpu_state(args.device), "model_revision":MODEL_REVISION}
    if args.delta == "fla":
        import fla, triton
        report.update({"fla":fla.__version__, "triton":triton.__version__})
    (directory/"benchmark.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["prepare", "native", "torch"], required=True)
    parser.add_argument("--output", type=Path, default=ROOT/"runs/bf16-comparison")
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--precision", choices=["fp32", "tf32x2", "tf32", "bf16"], default="bf16")
    parser.add_argument("--delta", choices=["reference", "fla"], default="reference")
    parser.add_argument("--head-chunk", type=int, default=2048)
    parser.add_argument("--vram-gib", type=float, default=None)
    parser.add_argument("--blas-workspace-mib", type=int, default=1024)
    parser.add_argument("--initial-adapter", type=Path, default=ROOT/"runs/optimizer-parity-final/muon/initial.f32")
    args = parser.parse_args()
    if not 8 <= args.context <= 4096:
        parser.error("Context range is 8..4096")
    args.total_batch = 65536 // max(args.context, 512)
    if args.batch is None:
        args.batch = min(args.total_batch, max(1, 8192 // args.context))
        while args.total_batch % args.batch:
            args.batch -= 1
    if args.vram_gib is None:
        args.vram_gib = 12
    if args.batch<1 or args.total_batch<1 or args.total_batch%args.batch or args.steps<=args.warmup or args.warmup<0:
        parser.error("Positive physical batch must divide total batch; steps must exceed nonnegative warmup")
    if args.backend == "torch" and args.precision == "tf32x2":
        parser.error("PyTorch modes are fp32, tf32 or bf16; corrected tf32x2 is native-only")
    path, ids, assets, info = shared_inputs(args)
    if args.backend == "native": native(args, path, info)
    elif args.backend == "torch": pytorch(args, ids, assets, info)
    else: print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
