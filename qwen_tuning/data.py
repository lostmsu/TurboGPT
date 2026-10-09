"""Bounded-memory tokenization and a chronological, record-aligned holdout."""

import hashlib
import json
import mmap
import re
from pathlib import Path

import numpy as np

from . import DATASET_ID, DEFAULT_CACHE, DEFAULT_TOKENS, MODEL_ID, MODEL_REVISION

RECORD_START = re.compile(rb"<[pc] id=\d+ ts=-?\d+")


def resolve_data(path: str | None, cache_dir: Path = DEFAULT_CACHE, offline: bool = False) -> Path:
    if path:
        result = Path(path).expanduser().resolve()
        if not result.is_file():
            raise FileNotFoundError(result)
        return result
    for candidate in (Path("hn1g.txt"), Path.home() / "Downloads" / "Datasets" / "hn1g.txt"):
        if candidate.is_file():
            return candidate.resolve()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        DATASET_ID, "hn1g.txt", repo_type="dataset", cache_dir=cache_dir,
        local_files_only=offline,
    ))


def source_hash(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining:
            chunk = stream.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise ValueError("Dataset changed while it was being read")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def split_offset(source: mmap.mmap, size: int, validation_fraction: float) -> int:
    target = int(size * (1 - validation_fraction))
    # Both current and older hn1g files have these record headers; bodies may
    # contain newlines, and older files have no LF between adjacent records.
    record = RECORD_START.search(source, target, size)
    if record:
        return record.start()
    # Support arbitrary plain UTF-8 text too, without cutting a code point.
    while target < size and source[target] & 0xC0 == 0x80:
        target += 1
    return target


def iter_chunks(source: mmap.mmap, start: int, end: int, chunk_bytes: int = 256 * 1024):
    while start < end:
        stop = min(start + chunk_bytes, end)
        if stop < end:
            match = RECORD_START.search(source, stop, min(stop + chunk_bytes, end))
            if match:
                stop = match.start()
            else:
                # End on whitespace if no record header is nearby, preserving
                # whitespace and UTF-8. Tokenization boundaries are reproducible.
                while stop < end and source[stop] not in b" \t\r\n":
                    stop += 1
                    if stop - start >= 2 * chunk_bytes:
                        break
                while stop < end and source[stop] & 0xC0 == 0x80:
                    stop += 1
        yield source[start:stop].decode("utf-8", errors="replace")
        start = stop


def prepare_data(
    path: Path,
    tokenizer,
    *,
    token_dir: Path = DEFAULT_TOKENS,
    validation_fraction: float = 0.01,
    max_bytes: int | None = None,
    revision: str = MODEL_REVISION,
) -> Path:
    if not 0 < validation_fraction < 0.5:
        raise ValueError("validation_fraction must be between 0 and 0.5")
    size = min(path.stat().st_size, max_bytes) if max_bytes else path.stat().st_size
    if size < 2:
        raise ValueError("Dataset must contain at least two bytes")
    metadata = {
        "format_version": 1, "source_sha256": source_hash(path, size),
        "source_bytes": size, "validation_fraction": validation_fraction,
        "model_id": MODEL_ID, "model_revision": revision,
        "tokenizer_class": type(tokenizer).__name__, "vocab_size": len(tokenizer),
        "add_special_tokens": False, "dtype": "uint32", "chunk_bytes": 256 * 1024,
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    destination = token_dir / key
    manifest = destination / "manifest.json"
    if manifest.exists():
        existing = json.loads(manifest.read_text())
        if all((destination / f"{name}.bin").stat().st_size == existing[f"{name}_tokens"] * 4
               for name in ("train", "validation")):
            print(f"Using token cache {destination}", flush=True)
            return destination
        raise ValueError(f"Incomplete token cache: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
        boundary = split_offset(source, size, validation_fraction)
        if boundary == 0 or boundary >= size:
            raise ValueError("Dataset is too small to create a validation split")
        metadata["split_byte_offset"] = boundary
        for name, start, end in (("train", 0, boundary), ("validation", boundary, size)):
            count = 0
            temporary = destination / f"{name}.bin.tmp"
            with temporary.open("wb") as output:
                for chunk_index, text in enumerate(iter_chunks(source, start, end)):
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    np.asarray(ids, dtype=np.uint32).tofile(output)
                    count += len(ids)
                    if chunk_index % 100 == 0:
                        print(f"Tokenizing {name}: {count:,} tokens", flush=True)
            temporary.replace(destination / f"{name}.bin")
            metadata[f"{name}_tokens"] = count
    metadata["source_path"] = str(path)
    temporary_manifest = manifest.with_suffix(".tmp")
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary_manifest.replace(manifest)
    print(f"Prepared {destination}: {metadata['train_tokens']:,} train / "
          f"{metadata['validation_tokens']:,} validation tokens", flush=True)
    return destination

