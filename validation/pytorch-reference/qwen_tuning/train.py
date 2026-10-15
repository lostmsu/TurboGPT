import importlib.metadata
import json
import math
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed

from . import MODEL_ID
from .data import PackedTokens
from .model import MemoryEfficientQwen

GIB = 1024**3


class MemoryReport(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            logs["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
            logs["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)


def train(args, tokenizer, tokens: Path):
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.resume:
        args.resume = args.resume.expanduser().resolve()
        if not (args.resume / "trainer_state.json").is_file() or not (args.resume / "optimizer.pt").is_file():
            raise ValueError("--resume must name a full Trainer checkpoint, including optimizer.pt")
        previous_config = args.resume.parent / "run_config.json"
        if not previous_config.is_file():
            raise ValueError("Resume checkpoint has no run_config.json in its parent directory")
        previous = json.loads(previous_config.read_text())
        dataset = json.loads((tokens / "manifest.json").read_text())
        identity_fields = ("format_version", "source_sha256", "source_bytes", "validation_fraction",
                           "model_id", "model_revision", "vocab_size", "add_special_tokens", "dtype",
                           "chunk_bytes", "split_byte_offset", "train_tokens", "validation_tokens")
        if any(previous["dataset"][name] != dataset[name] for name in identity_fields):
            raise ValueError("Resume dataset/tokenizer/split differs from the checkpoint run")
        for name in ("method", "revision", "sequence_length", "batch_size", "gradient_accumulation",
                     "rank", "alpha", "learning_rate", "seed"):
            if previous["arguments"][name] != getattr(args, name):
                raise ValueError(f"Resume --{name.replace('_', '-')} differs from the checkpoint run")
    elif args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is nonempty: {args.output_dir}. Use --resume or a new --output-dir.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required. Install with `uv sync` and check nvidia-smi.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires an Ampere or newer GPU with BF16 support.")
    total = torch.cuda.get_device_properties(0).total_memory
    if not 0 < args.vram_budget_gib < total / GIB:
        raise ValueError(f"VRAM budget must be below device capacity ({total / GIB:.2f} GiB)")
    torch.cuda.set_per_process_memory_fraction(args.vram_budget_gib * GIB / total, 0)
    torch.cuda.reset_peak_memory_stats()
    set_seed(args.seed)
    model_config = AutoConfig.from_pretrained(
        MODEL_ID, revision=args.revision, cache_dir=args.cache_dir,
        local_files_only=args.offline,
    ).text_config
    model_config.use_cache = False
    if args.sequence_length > model_config.max_position_embeddings:
        raise ValueError("Sequence length exceeds the model's context limit")
    load_options = {
        "config": model_config, "revision": args.revision, "cache_dir": args.cache_dir,
        "local_files_only": args.offline, "dtype": torch.bfloat16,
        "attn_implementation": "sdpa", "device_map": {"": 0},
        "output_loading_info": True,
        "key_mapping": {r"^model\.language_model\.": "model."},
    }
    if args.method == "qlora":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as error:
            raise RuntimeError("Install QLoRA dependencies with `uv sync --extra qlora`") from error
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_storage=torch.bfloat16,
        )
    model, loading = MemoryEfficientQwen.from_pretrained(MODEL_ID, **load_options)
    if loading.get("missing_keys") or loading.get("mismatched_keys"):
        raise RuntimeError(f"Pretrained text weights were not fully loaded: {loading}")
    model.loss_chunk_size = args.loss_chunk_size
    if args.method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=args.rank, lora_alpha=args.alpha,
        lora_dropout=0.0, target_modules="all-linear", bias="none", revision=args.revision,
    ))
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    adapted = [name for name, module in model.named_modules() if hasattr(module, "lora_A")]
    if not any("linear_attn" in name for name in adapted):
        raise RuntimeError("LoRA failed to cover Qwen's Gated DeltaNet layers")
    if any("visual" in name or "lm_head" in name for name in adapted):
        raise RuntimeError("Unexpected vision/output-head adapter")
    train_data = PackedTokens(tokens / "train.bin", args.sequence_length)
    eval_data = PackedTokens(tokens / "validation.bin", args.sequence_length, args.eval_blocks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "dataset": json.loads((tokens / "manifest.json").read_text()),
        "gpu": torch.cuda.get_device_name(), "device_capacity_gib": total / GIB,
        "vram_budget_gib": args.vram_budget_gib, "adapter_modules": adapted,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "peft", "accelerate", "huggingface-hub")},
        "effective_batch_tokens": args.batch_size * args.gradient_accumulation * (args.sequence_length - 1),
        "train_blocks": len(train_data), "eval_blocks": len(eval_data),
    }
    if args.method == "qlora":
        report["packages"]["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
    (args.output_dir / "run_config.json").write_text(json.dumps(report, indent=2) + "\n")
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate, weight_decay=0.01, max_grad_norm=1.0,
        lr_scheduler_type="cosine", warmup_steps=0.03, optim="adamw_torch",
        bf16=True, tf32=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        logging_steps=args.logging_steps, report_to="none", seed=args.seed, data_seed=args.seed,
        prediction_loss_only=True, dataloader_num_workers=0, dataloader_pin_memory=True,
        remove_unused_columns=False, label_names=["labels"],
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=eval_data,
        processing_class=tokenizer, callbacks=[MemoryReport()],
    )
    # PEFT exposes a generic forward signature; explicitly preserve mean-loss
    # normalization across gradient accumulation for this custom loss.
    trainer.model_accepts_loss_kwargs = False
    try:
        result = trainer.train(resume_from_checkpoint=str(args.resume) if args.resume else None)
        metrics = trainer.evaluate()
        final_checkpoint = args.output_dir / f"checkpoint-{trainer.state.global_step}"
        if not (final_checkpoint / "trainer_state.json").is_file():
            # The pinned Trainer has no public full-state checkpoint method.
            # Preserve optimizer/RNG state even when the final step is not a
            # multiple of save_steps; save_model + save_state alone cannot resume.
            trainer._save_checkpoint(trainer.model, trial=None)
        metrics.update(result.metrics)
        metrics["eval_perplexity"] = math.exp(min(metrics["eval_loss"], 80))
        metrics["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / GIB
        metrics["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / GIB
        metrics["vram_budget_gib"] = args.vram_budget_gib
        trainer.save_model(str(args.output_dir / "adapter"))
        trainer.save_state()
        trainer.save_metrics("all", metrics)
        print(json.dumps(metrics, indent=2))
    except torch.OutOfMemoryError as error:
        raise RuntimeError(
            f"Exceeded the {args.vram_budget_gib:g} GiB training budget or available VRAM. "
            "Reduce --sequence-length/--batch-size, or use --method qlora. "
            "Increase gradient accumulation to keep the effective batch size."
        ) from error
