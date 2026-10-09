"""CPU-only data preparation and token-budget statistics for native experiments."""

import json
import math
import shutil
from pathlib import Path

import numpy as np

from .data import source_hash


def sha256(path):
    return source_hash(Path(path), Path(path).stat().st_size)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def aligned_starts(tokens, lengths, byte_anchors, chunk_tokens=8_000_000):
    """First token boundary at/after each byte anchor, using bounded prefix sums.

    Positions refer to NFC-normalized UTF-8, as does Qwen's byte accounting.
    Return token indices, actual byte offsets, and the whole stream's byte count.
    """
    anchors = np.asarray(byte_anchors, dtype=np.uint64)
    if np.any(anchors[1:] < anchors[:-1]):
        raise ValueError("Byte anchors must be ordered")
    indices = np.zeros(len(anchors), dtype=np.uint64)
    actual = np.zeros(len(anchors), dtype=np.uint64)
    next_anchor = int(np.searchsorted(anchors, 0, side="right"))
    prefix = 0
    for offset in range(0, len(tokens), chunk_tokens):
        ends = np.cumsum(lengths[tokens[offset:offset + chunk_tokens]], dtype=np.uint64)
        if not len(ends):
            continue
        ends += prefix
        limit = int(np.searchsorted(anchors, ends[-1], side="right"))
        if limit > next_anchor:
            local = np.searchsorted(ends, anchors[next_anchor:limit], side="left")
            indices[next_anchor:limit] = offset + local + 1
            actual[next_anchor:limit] = ends[local]
            next_anchor = limit
        prefix = int(ends[-1])
    if next_anchor != len(anchors):
        raise ValueError("An anchor lies beyond the tokenized corpus")
    return indices, actual, prefix


def check_lanes(starts, token_count, context, steps):
    starts = [int(x) for x in starts]
    boundaries = starts[1:] + [int(token_count)]
    span = steps * context + 1  # Include the next-token target at the final edge.
    slack = [end - start - span for start, end in zip(starts, boundaries)]
    if any(x < 0 for x in slack) or any(a >= b for a, b in zip(starts, boundaries)):
        raise ValueError("A sequential lane crosses the next lane or corpus end")
    return {"span_tokens_including_target": span, "minimum_remaining_tokens": min(slack),
            "wraps": False, "crosses_next_start": False}


def prepare_full_corpus(asset_directory, destination, steps, contexts=(4096, 1024, 256)):
    """Join the verified historical split so every corpus token is trainable."""
    asset_directory, destination = Path(asset_directory), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    assets = json.loads((asset_directory / "assets.json").read_text())
    old = assets["dataset"]
    source = Path(old["source_path"])
    if source.stat().st_size != old["source_bytes"] or sha256(source) != old["source_sha256"]:
        raise ValueError("The full source dataset differs from the verified cache")
    parts = [Path(assets["paths"][key]) for key in ("train", "validation")]
    expected_bytes = 4 * (old["train_tokens"] + old["validation_tokens"])
    full = destination / "full.bin"
    if not full.exists():
        temporary = full.with_suffix(".tmp")
        with temporary.open("wb") as output:
            for part, count in zip(parts, (old["train_tokens"], old["validation_tokens"])):
                if part.stat().st_size != count * 4:
                    raise ValueError("Historical token cache has the wrong size")
                with part.open("rb") as stream:
                    shutil.copyfileobj(stream, output, 8 * 1024 * 1024)
        temporary.replace(full)
    if full.stat().st_size != expected_bytes:
        raise ValueError("Full token cache has the wrong size")
    full_hash = sha256(full)
    cached_manifest = destination / "assets.json"
    if cached_manifest.exists():
        recorded = json.loads(cached_manifest.read_text())
        if recorded["dataset"]["token_sha256"] != full_hash:
            raise ValueError("Full token cache changed")
    tokens = np.memmap(full, dtype="<u4", mode="r")
    lengths = np.fromfile(assets["paths"]["token_bytes"], dtype="<u4")
    largest = max(65_536 // context for context in contexts)
    anchors = np.arange(largest, dtype=np.uint64) * old["source_bytes"] // largest
    indices, actual, normalized_bytes = aligned_starts(tokens, lengths, anchors)
    dataset = {**old, "validation_fraction": 0, "held_out": False,
               "train_tokens": len(tokens), "validation_tokens": 0,
               "token_sha256": full_hash, "normalized_bytes": normalized_bytes,
               "cache_construction": "concatenated historical train and validation caches in source order",
               "historical_tokenization_boundary": old["split_byte_offset"]}
    dataset.pop("split_byte_offset")
    paths = {**assets["paths"], "train": str(full.resolve()), "validation": str(full.resolve())}
    # Validation points to the same file for the native path format; suite builds disable evaluation.
    (destination / "paths.txt").write_text("\n".join(paths.values()) + "\n", encoding="utf-8")
    atomic_json(destination / "assets.json", {**assets, "paths": paths, "dataset": dataset})
    for context in contexts:
        batch = 65_536 // context
        if largest % batch:
            raise ValueError("Expected nested lane counts")
        stride = largest // batch
        chosen = indices[::stride]
        safety = check_lanes(chosen, len(tokens), context, steps)
        chosen.astype("<u8").tofile(destination / f"seq-starts-{context}.u64")
        atomic_json(destination / f"seq-starts-{context}.json", {
            "context": context, "effective_batch": batch, "steps": steps,
            "corpus_bytes": old["source_bytes"], "corpus_tokens": len(tokens),
            "alignment": "first NFC-normalized token boundary at/after the byte anchor",
            "byte_anchor_formula": "actual_corpus_bytes * j // effective_batch",
            "normalized_bytes": normalized_bytes, "safety": safety,
            "lanes": [{"j": j, "requested_byte": int(a), "aligned_normalized_byte": int(b),
                       "token_index": int(c), "alignment_shift_bytes": int(b) - int(a)}
                      for j, (a, b, c) in enumerate(zip(anchors[::stride], actual[::stride], chosen))],
        })
    return dataset


def canonical_training_rows(path, max_step=None):
    """A restart supersedes the abandoned suffix after its recovered checkpoint."""
    rows = {}
    if not Path(path).exists():
        return []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # Only an interrupted final write can be incomplete.
        if row.get("kind") == "train" and (max_step is None or row["step"] <= max_step):
            rows[row["step"]] = row
    return [rows[key] for key in sorted(rows)]


def tail_average(rows, total_tokens, window_tokens):
    if not rows or window_tokens <= 0 or window_tokens > total_tokens:
        raise ValueError("Invalid final token window")
    boundary = total_tokens - window_tokens
    covered = 0
    weighted = 0.0
    nll = 0.0
    byte_count = 0.0
    updates = 0
    for row in rows:
        end, count = row["processed_tokens"], row["predicted_tokens"]
        included = max(0, min(end, total_tokens) - max(end - count, boundary))
        if included:
            fraction = included / count
            weighted += row["last_bpb"] * included
            nll += row["last_nll"] * fraction
            byte_count += row["last_bytes"] * fraction
            covered += included
            updates += 1
    if covered != window_tokens:
        raise ValueError(f"Final window contains {covered} tokens, expected {window_tokens}")
    return {"avg_last_bpb": weighted / covered, "pooled_last_bpb": nll / (math.log(2) * byte_count),
            "window_tokens": covered, "window_updates": updates,
            "window_start_exclusive": boundary, "window_end_inclusive": total_tokens}
