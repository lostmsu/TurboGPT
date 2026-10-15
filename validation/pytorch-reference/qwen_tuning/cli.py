import argparse
import os
from pathlib import Path

from . import DEFAULT_CACHE, DEFAULT_TOKENS, MODEL_ID, MODEL_REVISION, ROOT


def parser():
    result = argparse.ArgumentParser(description="Qwen3.5 LoRA on the raw hn1g text stream")
    result.add_argument("command", choices=["download", "prepare", "train"])
    result.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    result.add_argument("--revision", default=MODEL_REVISION)
    result.add_argument("--offline", action="store_true")
    result.add_argument("--data", help="hn1g.txt (default: local Downloads/Datasets, then HF Hub)")
    result.add_argument("--token-dir", type=Path, default=DEFAULT_TOKENS)
    result.add_argument("--validation-fraction", type=float, default=0.01)
    result.add_argument("--max-data-bytes", type=int, help="Limit preprocessing to a prefix, for smoke tests")
    result.add_argument("--method", choices=["lora", "qlora"], default="lora")
    result.add_argument("--device", default="0", help="Physical CUDA device index; expose exactly one GPU")
    result.add_argument("--vram-budget-gib", type=float, default=22.0)
    result.add_argument("--sequence-length", type=int, default=2048)
    result.add_argument("--batch-size", type=int, default=16)
    result.add_argument("--gradient-accumulation", type=int, default=1)
    result.add_argument("--rank", type=int, default=16)
    result.add_argument("--alpha", type=int, default=32)
    result.add_argument("--loss-chunk-size", type=int, default=128)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--epochs", type=float, default=1.0)
    result.add_argument("--max-steps", type=int, default=-1)
    result.add_argument("--eval-blocks", type=int, default=64)
    result.add_argument("--eval-steps", type=int, default=250)
    result.add_argument("--save-steps", type=int, default=250)
    result.add_argument("--logging-steps", type=int, default=10)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "qwen-hn1g")
    result.add_argument("--resume", type=Path, help="Trainer checkpoint directory, including optimizer/RNG state")
    return result


def main():
    args = parser().parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.token_dir = args.token_dir.expanduser().resolve()
    if args.offline:
        # PEFT also probes Hub metadata while saving adapters. Local-only model
        # loading alone does not prevent those requests.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    if args.command == "download":
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MODEL_ID, revision=args.revision, cache_dir=args.cache_dir,
            local_files_only=args.offline,
        )
        print(path)
        return
    for name in ("sequence_length", "batch_size", "gradient_accumulation", "rank", "alpha",
                 "loss_chunk_size", "eval_blocks", "eval_steps", "save_steps", "logging_steps"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_data_bytes is not None and args.max_data_bytes < 2:
        raise ValueError("--max-data-bytes must be at least 2")
    # Set these before torch, accelerate or transformers are imported.
    if args.command == "train":
        if "," in args.device:
            raise ValueError("Select one GPU with --device; this trainer is single-GPU")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    from transformers import AutoTokenizer
    from .data import prepare_data, resolve_data

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=args.revision, cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    tokenizer.model_max_length = 10**30  # Chunk tokenization precedes fixed-length packing.
    path = resolve_data(args.data, args.cache_dir, args.offline)
    tokens = prepare_data(
        path, tokenizer, token_dir=args.token_dir,
        validation_fraction=args.validation_fraction, max_bytes=args.max_data_bytes,
        revision=args.revision,
    )
    if args.command == "prepare":
        print(tokens)
        return
    from .train import train

    train(args, tokenizer, tokens)
