"""Extract the AFM `tokenizer` binary into a GGUF-ready vocab JSON.

FORMAT (reverse-engineered here; closes exactly, no slack bytes)
---------------------------------------------------------------
Apple ships the tokenizer as one flat little-endian blob:

    0x00  u32   0
    0x04  u32   ? (~end of string blob)
    0x08  u32   ? (offset inside string blob)
    0x0c  u32   0x29570   (same in every file -> trie/normalizer size)
    0x10  u32   0
    0x14  u32   0x40000001
    0x18  u32   4          (# of special-token definitions: unk,bos,eos,pad)
    0x1c  u32   7
    0x20  u32   V          <-- VOCAB SIZE
    0x24            u8  x V        token TYPE   (SentencePiece enum, see below)
    0x24 + 1*V      f32 x V        token SCORE
    0x24 + 5*V      u32 x (V+5)    byte offsets into the string blob
    0x24 + 9*V + 20 ...            NUL-separated UTF-8 string blob
    <after blob>                   Darts double-array trie (encoder side)

The offset array carries FIVE leading entries that are *not* vocab pieces --
they are the model's special-token declarations, in this order:

    idx 0 = " ⁇ "  (unk_surface)   idx 1 = <unk>   idx 2 = <bos>
    idx 3 = <eos>                       idx 4 = <pad>

so   piece(id) = string_at(offset[id + 5]).

That +5 shift is load-bearing.  Without it the table decodes to
['<unk>','<bos>','<eos>','<pad>',...] which *looks* plausible but is wrong by
4 ids for all 262144 tokens.  The shift is proved three ways:
  * type[0..2]==3 (CONTROL) and type[3]==2 (UNKNOWN) only agrees with
    <pad>,<eos>,<bos>,<unk> -- i.e. with the shift applied;
  * the 256 type-6 (BYTE) entries land exactly on ids 238..493, whose pieces
    then read <0x00>..<0xFF>;
  * the first type-1 (NORMAL) id, 494, reads "▁t" -- BPE merge rank 0.

Type enum is SentencePiece's and is already GGUF-compatible as-is:
    1 NORMAL   2 UNKNOWN   3 CONTROL   4 USER_DEFINED   5 UNUSED   6 BYTE

Scores are -(BPE merge rank): id 494 -> -0.0 ... id 255967 -> -255504.0,
zero everywhere else.  The vocab is stored in merge-rank order, which is why
the score array is monotone.
"""
import hashlib, json, struct
import numpy as np

# instruct_300m.tokenizer.generic == instruct_3b.tokenizer.generic (identical sha256)
SRC = ("/Volumes/D/FM_GenerativeModels_copy/purpose_auto/"
       "4300baeb93793037a5657892c32a2340e4094d3d.asset/AssetData/tokenizer")
OUT = "/Volumes/D/github/afm-ifp-teardown/local/nano_gguf/nano_tokenizer.json"

# Apple type -> what llama.cpp needs to actually *match* the piece.
# UNUSED(5) pieces are never matched by llama.cpp's SPM path, which would make
# the Gemma-style chat markers (<start_of_turn>/<end_of_turn>) untokenizable.
# ids 4..237 are the real named specials; 255968.. are genuine reserved slots.
NAMED_SPECIAL = range(4, 238)


def parse(path):
    b = open(path, "rb").read()
    V = struct.unpack_from("<I", b, 32)[0]
    types = np.frombuffer(b, dtype=np.uint8, offset=36, count=V)
    scores = np.frombuffer(b, dtype="<f4", offset=36 + V, count=V)
    offs = np.frombuffer(b, dtype="<u4", offset=36 + 5 * V, count=V + 5)
    blob = 36 + 9 * V + 20
    meta = [b[blob + int(offs[k]):b.index(b"\x00", blob + int(offs[k]))] for k in range(5)]
    pieces = []
    for i in range(V):
        o = blob + int(offs[i + 5])
        pieces.append(b[o:b.index(b"\x00", o)])
    # the blob must end exactly at the last piece -> layout closes
    end = blob + int(offs[V + 4]) + len(pieces[-1]) + 1
    return b, V, types, scores, pieces, meta, blob, end


def main():
    b, V, types, scores, pieces, meta, blob, end = parse(SRC)
    assert meta[0] == b" \xe2\x81\x87 " and meta[1:] == [b"<unk>", b"<bos>", b"<eos>", b"<pad>"]
    assert pieces[:4] == [b"<pad>", b"<eos>", b"<bos>", b"<unk>"]
    assert [i for i, t in enumerate(types) if t == 6] == list(range(238, 494))
    assert pieces[238] == b"<0x00>" and pieces[493] == b"<0xFF>"
    assert pieces[105] == b"<start_of_turn>" and pieces[106] == b"<end_of_turn>"
    assert len(set(pieces)) == V, "pieces are not unique"

    tgguf = types.copy()
    tgguf[list(NAMED_SPECIAL)] = 4  # USER_DEFINED so llama.cpp matches them verbatim

    doc = {
        "_comment": "AFM tokenizer decoded from Apple's flat `tokenizer` asset; see extract_nano_tokenizer.py for the format.",
        "source_asset": SRC,
        "source_asset_specifier": "com.apple.fm.language.instruct_300m.tokenizer.generic",
        "source_sha256": hashlib.sha256(b).hexdigest(),
        "applies_to": "afmplus-v11.0-nano (backbone_signature 15fcbb86c629cdd0c50598fe6f700108967f4a9c16ee462c2b90d18400b68567)",
        "vocab_size": V,
        "model": "llama",           # SentencePiece unigram; scores are -(merge rank)
        "add_bos": True,
        "add_eos": False,
        "bos_token_id": 2,
        "eos_token_id": 106,        # <end_of_turn>; raw <eos> is id 1
        "eot_token_id": 106,
        "eos_raw_token_id": 1,
        "pad_token_id": 0,
        "unk_token_id": 3,
        "unk_surface": " ⁇ ",
        "chat_start_of_turn_id": 105,
        "chat_end_of_turn_id": 106,
        "token_type_enum": {"1": "NORMAL", "2": "UNKNOWN", "3": "CONTROL",
                            "4": "USER_DEFINED", "5": "UNUSED", "6": "BYTE"},
        "tokens": [p.decode("utf-8", "surrogateescape") for p in pieces],
        "scores": [float(s) for s in scores],
        "token_type": [int(t) for t in types],       # verbatim from the asset
        "token_type_gguf": [int(t) for t in tgguf],  # ids 4..237 promoted 5 -> 4
    }
    with open(OUT, "w") as f:
        json.dump(doc, f, ensure_ascii=False)
    print("V=%d  blob=0x%x..0x%x (file 0x%x)  -> %s" % (V, blob, end, len(b), OUT))


if __name__ == "__main__":
    main()
