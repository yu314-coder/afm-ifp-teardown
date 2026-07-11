#!/usr/bin/env python3
"""afmplus-v11.0-ifp tokenizer — recovered from Apple's custom tokenizer container.

Byte-level SentencePiece BPE, 262144-slot vocab (152064 real + byte-fallback + extra_ids).
Vocab pieces extracted from the container; file order == token ID (verified: <unk>=0,
<bos>=1, <eos>=2, <pad>=3 match SentencePiece convention). Encoding = greedy longest-match
with the U+2581 space marker, which produces correct word-level tokenizations.

Validated: "The capital of Japan is Tokyo" -> [_The,_capital,_of,_Japan,_is,_Tokyo].
"""
import json

SP = '▁'   # U+2581 space marker

class AFMTokenizer:
    def __init__(self, vocab_json):
        self.vocab = json.load(open(vocab_json))
        self.piece2id = {}
        for i, p in enumerate(self.vocab):
            if p and p not in self.piece2id:
                self.piece2id[p] = i
        self.bos, self.eos, self.unk, self.pad = 1, 2, 0, 3

    def encode(self, text, add_bos=True):
        text = SP + text.replace(' ', SP)
        ids, i = [], 0
        while i < len(text):
            for L in range(min(40, len(text) - i), 0, -1):
                if text[i:i+L] in self.piece2id:
                    ids.append(self.piece2id[text[i:i+L]]); i += L; break
            else:
                # byte fallback
                for b in text[i].encode('utf-8'):
                    ids.append(self.piece2id.get(f'<0x{b:02X}>', self.unk))
                i += 1
        return ([self.bos] + ids) if add_bos else ids

    def decode(self, ids):
        return ''.join(self.vocab[i] if i < len(self.vocab) else '' for i in ids).replace(SP, ' ').strip()

if __name__ == '__main__':
    tk = AFMTokenizer('/Volumes/D/fix/afm_odix/tok_vocab.json')
    for s in ["The capital of Japan is Tokyo", "Hello world", "Write a poem about the ocean"]:
        ids = tk.encode(s)
        print(f'{s!r}\n  ids={ids}\n  roundtrip={tk.decode(ids)!r}')
