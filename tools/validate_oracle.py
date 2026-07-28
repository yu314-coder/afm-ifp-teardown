"""VALIDATE the residual-basis oracle against a known-good model.

Everything concluding "down's output ordering is wrong" rests on one unverified assumption:
that in a correct transformer, the per-channel magnitude with which a WRITER (attn_output,
ffn_down) writes residual position j correlates with the magnitude with which a READER
(ffn_gate/ffn_up) reads position j.

pico's data is equally consistent with that assumption being FALSE:
    readers agree with each other   +0.62      writers agree with each other  +0.49
    writer vs reader                ~0.00
If writer/reader magnitudes are simply uncorrelated in real models, then pico's writers are
not necessarily misordered at all and the oracle is measuring nothing.

Test on Qwen3-4B-Instruct (Q4_K_M) -- a real, correctly-ordered model in the SAME family as
pico (RMSNorm + per-head QK-norm + GQA + SwiGLU). If corr(attn_output cols, ffn_gate rows)
is large there, the oracle is sound. If it is ~0, the oracle is invalid and the down verdict
must be withdrawn.

Includes a Q4_K dequantizer (no gguf package available).
"""
import struct
import numpy as np

PATH = '/Volumes/D/github/image/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf'
GGML_F32, GGML_F16, GGML_Q4_K = 0, 1, 12


def read_gguf_index(path):
    f = open(path, 'rb')
    assert f.read(4) == b'GGUF'
    ver, ntens, nkv = struct.unpack('<IQQ', f.read(20))

    def rstr():
        n = struct.unpack('<Q', f.read(8))[0]
        return f.read(n).decode('utf-8', 'replace')

    def rval(t):
        if t == 8: return rstr()
        if t in (0, 1, 7): return f.read(1)
        if t in (2, 3): return f.read(2)
        if t in (4, 5, 6): return f.read(4)
        if t in (10, 11, 12): return f.read(8)
        if t == 9:
            et, cnt = struct.unpack('<IQ', f.read(12))
            for _ in range(cnt):
                rval(et)
            return None
        raise ValueError(t)

    kv = {}
    for _ in range(nkv):
        k = rstr(); t = struct.unpack('<I', f.read(4))[0]; v = rval(t)
        kv[k] = v
    tens = []
    for _ in range(ntens):
        name = rstr()
        nd = struct.unpack('<I', f.read(4))[0]
        dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
        typ, off = struct.unpack('<IQ', f.read(12))
        tens.append((name, dims, typ, off))
    align = 32
    pos = f.tell()
    data_start = pos + ((-pos) % align)
    return f, tens, data_start


def deq_q4k(buf, n):
    """Q4_K: 256 weights per super-block, 144 bytes."""
    nb = n // 256
    a = np.frombuffer(buf[:nb * 144], dtype=np.uint8).reshape(nb, 144)
    d = a[:, 0:2].copy().view(np.float16).astype(np.float32).ravel()
    dmin = a[:, 2:4].copy().view(np.float16).astype(np.float32).ravel()
    sc_raw = a[:, 4:16]
    qs = a[:, 16:144]
    # unpack 8 sub-block 6-bit scales and mins (llama.cpp get_scale_min_k4)
    sc = np.zeros((nb, 8), np.float32); mn = np.zeros((nb, 8), np.float32)
    for j in range(8):
        if j < 4:
            s = sc_raw[:, j] & 63
            m = sc_raw[:, j + 4] & 63
        else:
            s = (sc_raw[:, j + 4] & 15) | ((sc_raw[:, j - 4] >> 6) << 4)
            m = (sc_raw[:, j + 4] >> 4) | ((sc_raw[:, j] >> 6) << 4)
        sc[:, j] = s; mn[:, j] = m
    lo = (qs & 0xF).astype(np.float32)
    hi = (qs >> 4).astype(np.float32)
    out = np.zeros((nb, 256), np.float32)
    for j in range(4):                       # 4 pairs of 32-weight halves
        blk_lo, blk_hi = 2 * j, 2 * j + 1
        seg = slice(j * 32, (j + 1) * 32)
        out[:, blk_lo * 32:(blk_lo + 1) * 32] = (d * sc[:, blk_lo])[:, None] * lo[:, seg] - (dmin * mn[:, blk_lo])[:, None]
        out[:, blk_hi * 32:(blk_hi + 1) * 32] = (d * sc[:, blk_hi])[:, None] * hi[:, seg] - (dmin * mn[:, blk_hi])[:, None]
    return out.ravel()[:n]


GGML_Q6_K = 14


def deq_q6k(buf, n):
    """Q6_K: 256 weights per super-block, 210 bytes (ql[128], qh[64], int8 scales[16], d)."""
    nb = n // 256
    a = np.frombuffer(buf[:nb * 210], dtype=np.uint8).reshape(nb, 210)
    ql = a[:, 0:128]
    qh = a[:, 128:192]
    sc = a[:, 192:208].view(np.int8).astype(np.float32)
    d = a[:, 208:210].copy().view(np.float16).astype(np.float32).ravel()
    out = np.zeros((nb, 256), np.float32)
    for n2 in range(2):                       # two 128-weight halves
        qlh = ql[:, n2 * 64:(n2 + 1) * 64].astype(np.int16)
        qhh = qh[:, n2 * 32:(n2 + 1) * 32].astype(np.int16)
        for l in range(32):
            q1 = ((qlh[:, l] & 0xF) | (((qhh[:, l] >> 0) & 3) << 4)) - 32
            q2 = ((qlh[:, l + 32] & 0xF) | (((qhh[:, l] >> 2) & 3) << 4)) - 32
            q3 = ((qlh[:, l] >> 4) | (((qhh[:, l] >> 4) & 3) << 4)) - 32
            q4 = ((qlh[:, l + 32] >> 4) | (((qhh[:, l] >> 6) & 3) << 4)) - 32
            base = n2 * 128
            for k, q in enumerate((q1, q2, q3, q4)):
                idx = base + k * 32 + l
                out[:, idx] = d * sc[:, (idx // 16)] * q
    return out.ravel()[:n]


def load(f, data_start, entry):
    name, dims, typ, off = entry
    n = int(np.prod(dims))
    f.seek(data_start + off)
    if typ == GGML_F32:
        raw = np.frombuffer(f.read(n * 4), dtype=np.float32)
    elif typ == GGML_F16:
        raw = np.frombuffer(f.read(n * 2), dtype=np.float16).astype(np.float32)
    elif typ == GGML_Q4_K:
        raw = deq_q4k(f.read((n // 256) * 144), n)
    elif typ == GGML_Q6_K:
        raw = deq_q6k(f.read((n // 256) * 210), n)
    else:
        return None
    return raw.reshape(dims[1], dims[0])      # [ne1, ne0] = [out, in]


def c(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


f, tens, ds = read_gguf_index(PATH)
byname = {t[0]: t for t in tens}
nl = max(int(n.split('.')[1]) for n, *_ in tens if n.startswith('blk.')) + 1
print('Qwen3-4B: %d tensors, %d layers' % (len(tens), nl))
print()
print('%-5s %-18s %-18s %-18s' % ('L', 'O_out~gate_in', 'down_out~gate_in', 'ref gate_in~up_in'))
A, B, R = [], [], []
for L in range(0, min(nl, 12)):
    try:
        O = load(f, ds, byname['blk.%d.attn_output.weight' % L])    # [out=n_embd, in=n_embd]
        G = load(f, ds, byname['blk.%d.ffn_gate.weight' % L])       # [out=n_ff, in=n_embd]
        U = load(f, ds, byname['blk.%d.ffn_up.weight' % L])
        D = load(f, ds, byname['blk.%d.ffn_down.weight' % L])       # [out=n_embd, in=n_ff]
    except KeyError:
        continue
    o_out = np.linalg.norm(O, axis=1)      # per OUTPUT (residual)
    d_out = np.linalg.norm(D, axis=1)      # per OUTPUT (residual)
    g_in = np.linalg.norm(G, axis=0)       # per INPUT  (residual)
    u_in = np.linalg.norm(U, axis=0)
    a, b, r = c(o_out, g_in), c(d_out, g_in), c(g_in, u_in)
    A.append(a); B.append(b); R.append(r)
    print('%-5d %+.4f            %+.4f            %+.4f' % (L, a, b, r))

print()
print('MEAN  O_out~gate_in    %+.4f' % np.mean(A))
print('MEAN  down_out~gate_in %+.4f' % np.mean(B))
print('MEAN  gate_in~up_in    %+.4f  (reader-reader reference)' % np.mean(R))
print()
if abs(np.mean(A)) > 0.30 and abs(np.mean(B)) > 0.30:
    print('=> ORACLE IS SOUND: in a correct model, writers DO track readers.')
    print('   pico\'s writer~reader ~0 therefore indicates a genuine ordering defect.')
else:
    print('=> ORACLE IS INVALID: even in a correct model, writer and reader magnitudes')
    print('   do NOT track each other. pico\'s writer~reader ~0 is then EXPECTED, not a')
    print('   defect, and every conclusion drawn from that oracle must be withdrawn.')
