"""Download and prepare native trainer assets. No PyTorch dependency."""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen_tuning import DEFAULT_CACHE, MODEL_ID, MODEL_REVISION, ROOT
from qwen_tuning.data import prepare_data, resolve_data, source_hash
from qwen_tuning.tokenizer import RawTokenizer, byte_lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--data")
    parser.add_argument("--tokens", type=Path, help="Reuse an existing verified token-cache directory")
    parser.add_argument("--output", type=Path, default=ROOT / ".cache" / "native")
    args = parser.parse_args()
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=DEFAULT_CACHE,
                                      local_files_only=args.offline))
    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    text = config["text_config"]
    assert (text["hidden_size"], text["num_hidden_layers"], text["intermediate_size"],
            text["vocab_size"]) == (1024, 24, 3584, 248320)
    tokenizer = RawTokenizer(snapshot / "tokenizer.json")
    data = resolve_data(args.data, DEFAULT_CACHE, args.offline)
    if args.tokens:
        tokens = args.tokens.resolve()
        manifest = json.loads((tokens / "manifest.json").read_text())
        assert manifest["source_sha256"] == source_hash(data, manifest["source_bytes"])
        cached = np.fromfile(tokens / "train.bin", dtype=np.uint32, count=5000)
        # Verify byte-level tokenization agrees with the old cache on a complete
        # prefix; truncate before the final potentially incomplete BPE token.
        with data.open("rb") as source:
            prefix = source.read(20000).decode("utf-8", errors="replace")
        expected = tokenizer.encode(prefix)
        np.testing.assert_array_equal(cached[:4000], np.asarray(expected[:4000]))
    else:
        tokens = prepare_data(data, tokenizer)
        manifest = json.loads((tokens / "manifest.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    weights = snapshot / "model.safetensors-00001-of-00001.safetensors"
    with weights.open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
    rows = []
    for name, entry in sorted(header.items()):
        if name.startswith("model.language_model."):
            name = name.removeprefix("model.language_model.")
            shape = entry["shape"]
            rows.append(f"{name} {entry['dtype']} {8 + header_size + entry['data_offsets'][0]} "
                        f"{len(shape)} " + " ".join(map(str, shape)))
    (args.output / "weights.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    lengths = byte_lengths(tokenizer.backend)
    lengths.tofile(args.output / "token_bytes.bin")
    paths = {"weights": str(weights.resolve()), "manifest": str((args.output / "weights.tsv").resolve()),
             "train": str((tokens / "train.bin").resolve()),
             "validation": str((tokens / "validation.bin").resolve()),
             "token_bytes": str((args.output / "token_bytes.bin").resolve())}
    (args.output / "paths.txt").write_text("\n".join(paths.values()) + "\n", encoding="utf-8")
    (args.output / "assets.json").write_text(json.dumps({"paths": paths, "dataset": manifest,
                                                        "model_revision": MODEL_REVISION}, indent=2) + "\n")
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()
