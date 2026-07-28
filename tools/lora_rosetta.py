"""Recover the OutTrans=1 intra-bank coefficient order from the LoRA constant data.

WHY THIS WORKS (the Rosetta stone):
  hwx segment __MKERN_0 has VM size 0x1c50000 = 29,687,808 and File Size 0 -- an
  unbacked, runtime-patched mutable kernel. lora_32_constant_data.bin is EXACTLY
  29,687,808 bytes (and __MKERN_9 = 0x38a0000 = 59,375,616 = lora_64_constant_data.bin
  exactly). So the constant-data file is DMA'd verbatim into ANE coefficient memory:
  its bytes ARE the ANE coefficient stream, already in tile order.

  Crucially the LoRA tensors are UNPALETTIZED fp16 (task 157: no Pal field;
  CoeffSize 0x1000 x 16 banks = 65536 B for 32x1024 = exactly 2 B/elem), and the
  32->1024 up-projections are the OutTrans=1 ones (census: 36 of them), while the
  1024->32 down-projections are OutTrans=0 (111). Same file, both modes, no
  quantization loss -- so the intra-bank order can be read directly.

FILE LAYOUT (derived exactly, no residual):
  14,843,904 fp16 = 24 layers x 618,496, and per layer the rank-32 adapters for all
  seven roles sum to exactly 618,496:
     Q  A[1024,32]+B[32,1024] = 65536      O    A[1024,32]+B[32,1024] = 65536
     K  A[1024,32]+B[32, 256] = 40960      gate A[1024,32]+B[32,3200] = 135168
     V  A[1024,32]+B[32, 256] = 40960      up   A[1024,32]+B[32,3200] = 135168
                                           down A[3200,32]+B[32,1024] = 135168

THE TEST:
  A 32->1024 tensor is stored as 16 banks x 2048 fp16 = 64 outputs x 32 ranks per bank.
  The RANK axis is GLOBAL (the same 32 latent directions in every bank); the OUTPUT axis
  is LOCAL (different 64 channels per bank). So under the CORRECT intra-bank order, the
  per-rank norm profile extracted from each bank is the same quantity in every bank and
  correlates strongly across banks. Under the WRONG order the extracted profile mixes
  output channels and decorrelates. That asymmetry identifies the order.
"""
import numpy as np

P = ('/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto/'
     '031c7be6f8fddbff0a6650fee75e345b1ee9613c.asset/.AssetData/model.odixpackage/MPSGraph/'
     'mpsExecutable.mpsgraphpackage/lora_32_constant_data.bin')

R = 32
PER_LAYER = 618496
raw = np.fromfile(P, dtype=np.uint8)
v = np.frombuffer(raw.tobytes(), dtype=np.float16).astype(np.float32)
print('lora_32_constant_data.bin: %d fp16, %d layers x %d' % (v.size, v.size // PER_LAYER, PER_LAYER))
assert v.size == 24 * PER_LAYER

# role table: (name, A_elems, B_elems, B_out)   A=[in,32]  B=[32,out]
ROLES = [('Q', 1024 * R, R * 1024, 1024),
         ('K', 1024 * R, R * 256, 256),
         ('V', 1024 * R, R * 256, 256),
         ('O', 1024 * R, R * 1024, 1024),
         ('gate', 1024 * R, R * 3200, 3200),
         ('up', 1024 * R, R * 3200, 3200),
         ('down', 3200 * R, R * 1024, 1024)]
assert sum(a + b for _, a, b, _ in ROLES) == PER_LAYER


def profile_consistency(blk, out_dim, nbanks=16):
    """blk: flat fp16 for a [32, out_dim] tensor. Returns (corr_in_fast, corr_out_fast).

    Splits into nbanks equal banks; each bank holds out_dim/nbanks outputs x 32 ranks.
    For each hypothesis, extract a length-32 per-rank norm profile per bank, then report
    the mean pairwise correlation of those profiles across banks.
    """
    per = blk.size // nbanks
    o_local = out_dim // nbanks
    assert per == o_local * R, (per, o_local, R)
    prof_if, prof_of = [], []
    for b in range(nbanks):
        w = blk[b * per:(b + 1) * per]
        # hypothesis "in_fast": index = o_local*R + r  -> [o_local, R], rank = columns
        a = w.reshape(o_local, R)
        prof_if.append(np.linalg.norm(a, axis=0))
        # hypothesis "out_fast": index = r*o_local + o  -> [R, o_local], rank = rows
        c = w.reshape(R, o_local)
        prof_of.append(np.linalg.norm(c, axis=1))

    def meancorr(ps):
        ps = [p - p.mean() for p in ps]
        cs = []
        for i in range(len(ps)):
            for j in range(i + 1, len(ps)):
                d = np.linalg.norm(ps[i]) * np.linalg.norm(ps[j])
                if d > 0:
                    cs.append(float(ps[i] @ ps[j] / d))
        return float(np.mean(cs))
    return meancorr(prof_if), meancorr(prof_of)


print()
print('=== per-rank profile consistency across banks (32->out B matrices) ===')
print('%-6s %-6s %-14s %-14s %s' % ('layer', 'role', 'in_fast', 'out_fast', 'verdict'))
tally = {'in_fast': 0, 'out_fast': 0}
for L in range(6):
    base = L * PER_LAYER
    off = base
    for name, na, nb, bout in ROLES:
        off += na                                   # skip A
        blk = v[off:off + nb]
        off += nb
        if bout not in (1024, 3200):                # need >=16 outputs per bank, keep it clean
            continue
        try:
            cif, cof = profile_consistency(blk, bout)
        except AssertionError:
            continue
        win = 'in_fast' if cif > cof else 'out_fast'
        tally[win] += 1
        print('%-6d %-6s %+.4f        %+.4f        %s' % (L, name, cif, cof, win))
print()
print('tally:', tally)
