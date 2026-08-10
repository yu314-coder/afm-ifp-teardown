#!/usr/bin/env python3
"""
Parser for Apple's on-device tokenizer asset (`AssetData/tokenizer`).

The file is NOT a SentencePiece protobuf. It is a flat little-endian blob whose
layout was recovered by structural analysis; the layout below closes with zero
slack for V = 262144 and V = 328192.

    0x00            u32     0
    0x0c            u32     0x29570         (constant across every asset seen)
    0x18            u32     4               (number of special-token declarations)
    0x20            u32     V               vocabulary size
    0x24            u8  x V                 token type (SentencePiece enum)
    0x24 + 1*V      f32 x V                 token score
    0x24 + 5*V      u32 x (V+5)             offsets, relative to the string blob
    0x24 + 9*V + 20 ...                     NUL-separated UTF-8 string blob
    <after blob>                            Darts double-array trie (encoder side)

THE +5 SHIFT IS LOAD-BEARING:

    piece(id) = string_at(BLOB + offset[id + 5])

The offset array carries five leading entries that are the model's special-token
*declarations*, not vocabulary:

    [" ⁇ " (unk_surface), "<unk>", "<bos>", "<eos>", "<pad>"]

Three independent structural checks fix the shift at exactly +5:

  1. type[0..2] == 3 (CONTROL) and type[3] == 2 (UNKNOWN). Only +5 puts
     <pad>,<eos>,<bos> on the CONTROL ids and <unk> on the UNKNOWN id.
  2. The 256 type-6 (BYTE) entries occupy ids 238..493. Only +5 makes those read
     as a contiguous <0x00>..<0xFF>; at +1 they read "</li>".."<0xFB>".
  3. The first NORMAL id, 494, reads "▁t" -- BPE merge rank 0.

Scores are 0.0 for ids 0..493, then -(BPE merge rank) for the NORMAL range, so
the vocabulary is stored in merge-rank order.

    !! An earlier scrape in this project (local/afm_odix/tok_vocab.json) used a
    !! shift of +1 and is therefore OFF BY FOUR. See TOKENIZER_CORRECTION.md.

No Apple data is contained in this file -- it reads the asset already on the
machine.
"""
import numpy as np


class AFMTokenizerAsset:
    def __init__(self, path):
        self.raw = open(path, 'rb').read()
        b = self.raw
        self.n_special = int(np.frombuffer(b, np.uint32, 1, 0x18)[0])
        V = self.vocab_size = int(np.frombuffer(b, np.uint32, 1, 0x20)[0])
        self.token_type = np.frombuffer(b, np.uint8, V, 0x24).copy()
        self.scores = np.frombuffer(b, np.float32, V, 0x24 + V).copy()
        self._off = np.frombuffer(b, np.uint32, V + 5, 0x24 + 5 * V)
        self._blob = 0x24 + 9 * V + 20
        self.SHIFT = 5

    def _at(self, k):
        o = self._blob + int(self._off[k])
        return self.raw[o:self.raw.index(b'\0', o)].decode('utf-8', 'surrogateescape')

    def declarations(self):
        """The five leading entries: unk_surface, <unk>, <bos>, <eos>, <pad>."""
        return [self._at(k) for k in range(self.SHIFT)]

    def piece(self, tid):
        return self._at(tid + self.SHIFT)

    def tokens(self):
        return [self._at(t + self.SHIFT) for t in range(self.vocab_size)]

    def gguf_token_type(self):
        """Apple's enum is already GGUF-compatible; promote the named specials
        (ids 4..237) from UNUSED(5) to USER_DEFINED(4) so llama.cpp emits them
        verbatim rather than treating them as unused slots."""
        t = self.token_type.copy()
        t[4:238][t[4:238] == 5] = 4
        return t

    def check(self):
        """Re-derive the +5 shift from the file itself. Raises on mismatch."""
        byte_ids = np.where(self.token_type == 6)[0]
        assert len(byte_ids) == 256, f'expected 256 BYTE tokens, got {len(byte_ids)}'
        assert self.piece(byte_ids.min()) == '<0x00>', 'BYTE class misaligned'
        assert self.piece(byte_ids.max()) == '<0xFF>', 'BYTE class misaligned'
        assert self.token_type[3] == 2 and self.piece(3) == '<unk>', 'UNKNOWN misplaced'
        assert self.piece(byte_ids.max() + 1) == '▁t', 'first NORMAL is not merge rank 0'
        return True


if __name__ == '__main__':
    import sys
    tk = AFMTokenizerAsset(sys.argv[1])
    tk.check()
    print(f'V={tk.vocab_size}  declarations={tk.declarations()}')
    for tid in (0, 1, 2, 3, 105, 106, 238, 493, 494):
        print(f'  {tid:6d} {tk.piece(tid)!r:20} type={tk.token_type[tid]} '
              f'score={tk.scores[tid]}')
    print('self-check passed: +5 shift re-derived from the asset')
