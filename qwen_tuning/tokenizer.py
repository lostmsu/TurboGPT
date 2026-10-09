"""Direct Rust-tokenizers loading and byte accounting for native CUDA training."""
import json

import numpy as np
from tokenizers import Tokenizer


class RawTokenizer:
    def __init__(self, path):
        self.backend = Tokenizer.from_file(str(path))

    def __len__(self):
        return self.backend.get_vocab_size()

    def encode(self, text, add_special_tokens=False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens).ids


def byte_lengths(tokenizer: Tokenizer, vocabulary_size=248320):
    decoder = json.loads(tokenizer.decoder.__getstate__())
    if decoder["type"] != "ByteLevel":
        raise ValueError("BPB requires this model's byte-level BPE decoder")
    lengths = np.zeros(vocabulary_size, dtype=np.uint32)
    # A ByteLevel vocabulary symbol represents ONE tokenizer-normalized byte, including
    # tokens ending inside UTF-8 characters. decode(id) would replace partial
    # characters and inflate their byte lengths.
    for piece, index in tokenizer.get_vocab(with_added_tokens=False).items():
        lengths[index] = len(piece)
    for piece, index in tokenizer.get_vocab(with_added_tokens=True).items():
        if not lengths[index]:
            lengths[index] = len(piece.encode("utf-8"))
    return lengths
