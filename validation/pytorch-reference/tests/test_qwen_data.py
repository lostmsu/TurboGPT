import json
import mmap

import numpy as np

from qwen_tuning.data import PackedTokens, iter_chunks, prepare_data, split_offset


class ByteTokenizer:
    def __len__(self):
        return 256

    def encode(self, text, add_special_tokens):
        assert not add_special_tokens
        return list(text.encode("utf-8"))


def test_record_split_and_chunks_preserve_multiline_utf8(tmp_path):
    records = [f"<c id={i} ts={i}>hello\n\n世界 café {i}</c>" for i in range(100)]
    raw = "".join(records).encode()
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


def test_packing_covers_every_next_token_pair_once(tmp_path):
    path = tmp_path / "tokens.bin"
    np.arange(13, dtype=np.uint32).tofile(path)
    dataset = PackedTokens(path, sequence_length=5)
    assert len(dataset) == 3
    pairs = []
    for index in range(len(dataset)):
        example = dataset[index]
        assert example["labels"].tolist() == example["input_ids"].tolist()
        pairs.extend(zip(example["input_ids"][:-1].tolist(), example["labels"][1:].tolist()))
    assert pairs == [(index, index + 1) for index in range(12)]
    assert len(PackedTokens(path, sequence_length=5, max_blocks=2)) == 2
