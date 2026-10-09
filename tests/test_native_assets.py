import json
import mmap

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.pre_tokenizers import ByteLevel

from qwen_tuning.data import iter_chunks, prepare_data, split_offset
from qwen_tuning.tokenizer import byte_lengths


class ByteTokenizer:
    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def test_chronological_split_preserves_multiline_records_and_utf8(tmp_path):
    raw = "".join(f"<c id={i} ts={i}>hello\n\n世界 café {i}</c>" for i in range(100)).encode()
    path = tmp_path / "hn1g.txt"
    path.write_bytes(raw)
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
        split = split_offset(source, len(raw), 0.1)
        assert raw[split:].startswith(b"<c id=")
        train = "".join(iter_chunks(source, 0, split, chunk_bytes=47)).encode()
        validation = "".join(iter_chunks(source, split, len(raw), chunk_bytes=47)).encode()
        assert train + validation == raw
        assert train.endswith(b"</c>")
    cache = prepare_data(path, ByteTokenizer(), token_dir=tmp_path / "tokens", validation_fraction=0.1)
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["split_byte_offset"] == split
    assert np.fromfile(cache / "train.bin", dtype=np.uint32).tolist() == list(train)
    assert np.fromfile(cache / "validation.bin", dtype=np.uint32).tolist() == list(validation)
    assert prepare_data(path, ByteTokenizer(), token_dir=tmp_path / "tokens", validation_fraction=0.1) == cache
    path.write_bytes(raw.replace(b"hello", b"HELLO"))
    assert prepare_data(path, ByteTokenizer(), token_dir=tmp_path / "tokens", validation_fraction=0.1) != cache


def test_plain_text_chunks_do_not_cut_utf8(tmp_path):
    raw = ("世界🌍 café\n" * 100).encode()
    path = tmp_path / "text.txt"
    path.write_bytes(raw)
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as source:
        split = split_offset(source, len(raw), 0.13)
        pieces = list(iter_chunks(source, 0, split, 7)) + list(iter_chunks(source, split, len(raw), 7))
        assert "".join(pieces).encode() == raw


def test_bpb_counts_bytes_inside_partial_utf8_tokens():
    # An unmerged byte vocabulary forces each UTF-8 byte into its own token.
    # Decoding a token by itself would produce replacement characters.
    vocab = {symbol: index for index, symbol in enumerate(sorted(ByteLevel.alphabet()))}
    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(["<特殊>"])
    lengths = byte_lengths(tokenizer, vocabulary_size=300)
    text = "世界🌍 café\n"
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    assert lengths[ids].sum() == len(text.encode("utf-8"))
    assert max(lengths[ids]) == 1
    assert lengths[tokenizer.token_to_id("<特殊>")] == len("<特殊>".encode("utf-8"))
    assert np.all(lengths[257:] == 0)


def test_full_model_byte_accounting_when_cached():
    # This check uses only tokenizer JSON; it does not load neural weights.
    from qwen_tuning import DEFAULT_CACHE, MODEL_REVISION
    path = DEFAULT_CACHE / "models--Qwen--Qwen3.5-0.8B-Base" / "snapshots" / MODEL_REVISION / "tokenizer.json"
    if not path.exists():
        import pytest
        pytest.skip("Optional local tokenizer snapshot not present")
    tokenizer = Tokenizer.from_file(str(path))
    lengths = byte_lengths(tokenizer)
    for text in ["Hello, Hacker News!\n", "世界🌍 café — naïve", "<c id=123 ts=42>some\ntext</c>", "cafe\u0301"]:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        normalized = tokenizer.normalizer.normalize_str(text)
        assert lengths[ids].sum() == len(normalized.encode("utf-8"))


def test_sequential_byte_alignment_and_boundary_checks():
    from qwen_tuning.experiments import aligned_starts, check_lanes
    lengths = np.array([1, 3, 2, 5], dtype=np.uint32)
    tokens = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.uint32)
    indices, actual, total = aligned_starts(tokens, lengths, [0, 2, 6, 11], chunk_tokens=2)
    assert indices.tolist() == [0, 2, 3, 4]
    assert actual.tolist() == [0, 4, 6, 11]
    assert total == 22
    assert check_lanes([0, 4], 8, context=1, steps=3)["minimum_remaining_tokens"] == 0
    import pytest
    with pytest.raises(ValueError, match="crosses"):
        check_lanes([0, 4], 8, context=1, steps=4)


def test_final_bpb_window_uses_equal_tokens_despite_different_durations():
    from qwen_tuning.experiments import tail_average
    rows = [{"processed_tokens": (i+1)*100, "predicted_tokens": 100, "last_bpb": float(i+1),
             "last_nll": (i+1)*np.log(2)*10, "last_bytes": 10, "update_seconds": 1 if i < 3 else 999}
            for i in range(4)]
    result = tail_average(rows, 400, 200)
    assert result["avg_last_bpb"] == 3.5
    assert np.isclose(result["pooled_last_bpb"], 3.5)
    assert result["window_updates"] == 2
    assert result["window_start_exclusive"] == 200
