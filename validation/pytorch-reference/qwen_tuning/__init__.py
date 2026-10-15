"""Qwen fine-tuning alongside the original CUDA Rust experiments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / ".cache" / "huggingface" / "hub"
DEFAULT_TOKENS = ROOT / ".cache" / "tokens"
MODEL_ID = "Qwen/Qwen3.5-0.8B-Base"
MODEL_REVISION = "dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68"
DATASET_ID = "lostmsu/hn1g"
DATASET_SHA256 = "000beb89de92941829a2f64ee3f0f500de92c98959dd47e97b9a7347b9b285b9"
