#!/usr/bin/env python3
"""afm2gguf — build a llama.cpp GGUF (architecture `afm`) of Apple's on-device foundation model
(afmplus-v11.2-ifp, `SystemLanguageModel`) from the model assets already installed on THIS Mac.

Requirements: macOS 27 with the Apple Intelligence language-model assets downloaded, Python 3.9+,
numpy, gguf (`pip install numpy gguf`). The output runs with llama.cpp built with the `afm` architecture patch.

This script contains no Apple data: every weight, scale, table and string is read from the local assets.
The weights are written exactly — Apple's 2-bit codes + fp16 row scales as Q2_K (sc=2, m=3, d=dmin=s/2 ⇒
d·sc·c − dmin·m = s·(c − 1.5), bit-exact) and the int4 embedding as Q4_0 (exact).

usage: python3 afm2gguf.py -o afm-v11.2-ifp.gguf [--asset DIR] [--tokenizer FILE] [--no-kvq]
       python3 afm2gguf.py --list-adapters
       python3 afm2gguf.py --adapter summarization -o summarization.lora.gguf   (an Apple Intelligence feature adapter)
       python3 afm2gguf.py --draft -o afm-v11.2-draft.gguf   (the speculative-decoding draft model, arch afm-draft)
       python3 afm2gguf.py --pico -o afm-300m-base.gguf     (the 300M base model, arch afm-draft)
       python3 afm2gguf.py --pico-adapter safety -o afm-300m-safety.lora.gguf   (a 300M adapter, e.g. the safety guard)
"""
import argparse, glob, json, mmap, os, plistlib, re, struct, sys, time
import numpy as np
try:
    import gguf
    from gguf import GGMLQuantizationType as QT
except ImportError:
    sys.exit('error: the `gguf` Python package is required (pip install gguf)')

ASSETS = '/System/Library/AssetsV2/com_apple_MobileAsset_UAF_FM_GenerativeModels/purpose_auto'
SPEC_MODEL = 'com.apple.fm.language.instruct_3b.base.generic_sparse'
SPEC_TOK = 'com.apple.fm.language.instruct_3b.tokenizer.generic_sparse'
SPEC_API = 'com.apple.fm.language.instruct_3b.fm_api_generic.generic_sparse'
DISPLAY = 'afmplus-v11.2-ifp'

# architecture (afmplus-v11.2-ifp); every value is checked against the asset while decoding
NL, D, NH, NKV, HD = 56, 1536, 16, 8, 128
NDENSE, NSTD, NREUSE, KVSRC = 12, 23, 21, 34
FF, NE, ES, NSHARED, NMASK, NROUTED = 3072, 380, 256, 23, 32, 23
PAL = np.array([-1.5, -0.5, 0.5, 1.5], np.float32)
D_DRAFT = 256

def log(*a): print('[afm2gguf]', *a, flush=True)

# ----------------------------------------------------------------------------- asset discovery
def find_asset(spec):
    for a in sorted(glob.glob(f'{ASSETS}/*.asset')):
        try:
            d = plistlib.load(open(f'{a}/Info.plist', 'rb'))
            if (d.get('MobileAssetProperties') or {}).get('AssetSpecifier', d.get('AssetSpecifier')) == spec: return a
        except Exception: pass
    return None

def locate(args):
    root = args.asset
    if root is None:
        a = find_asset(SPEC_MODEL)
        if a is None: sys.exit(f'error: asset {SPEC_MODEL} not found under {ASSETS} (is Apple Intelligence enabled and downloaded?)')
        root = f'{a}/.AssetData' if os.path.isdir(f'{a}/.AssetData/model.odixpackage') else None
        if root is None:
            dmg = glob.glob(f'{a}/AssetData/Restore/*.dmg')
            sys.exit('error: model files are not mounted. Mount the cryptex read-only and pass --asset:\n'
                     f'  hdiutil attach -readonly -nobrowse -noverify -mountpoint /tmp/afm "{dmg[0] if dmg else "<dmg>"}"\n'
                     '  python3 afm2gguf.py --asset /tmp/afm -o afm.gguf')
    meta = json.load(open(f'{root}/metadata.json'))
    if meta.get('display_version') != DISPLAY:
        sys.exit(f"error: this converter supports {DISPLAY}, the asset is {meta.get('display_version')}")
    pkg = f'{root}/model.odixpackage'
    P = dict(meta=meta, hwx=f'{pkg}/main-h16g-delegates/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx',
             odix=f'{pkg}/main-h16g.odix', ifp=f'{pkg}/ifp/ifp_rasterized_weights.bin',
             ifpcfg=f'{pkg}/ifp/config.json', router=f'{pkg}/ifp/project_experts.mlasset/main-unspecialized.odix')
    tok = args.tokenizer
    if tok is None:
        t = find_asset(SPEC_TOK)
        tok = f'{t}/AssetData/tokenizer' if t else None
    if not tok or not os.path.exists(tok): sys.exit(f'error: tokenizer asset {SPEC_TOK} not found; pass --tokenizer')
    P['tok'] = tok
    api = find_asset(SPEC_API)
    P['api_meta'] = json.load(open(f'{api}/AssetData/metadata.json')) if api and os.path.exists(f'{api}/AssetData/metadata.json') else {}
    for k in ('hwx', 'odix', 'ifp', 'ifpcfg', 'router'):
        if not os.path.exists(P[k]): sys.exit(f'error: missing {P[k]}')
    return P

# ----------------------------------------------------------------------------- Mach-O (hwx) parsing
def macho(path):
    b = open(path, 'rb').read(1 << 20); n = struct.unpack_from('<I', b, 16)[0]; o = 32; seg, sect = {}, {}
    for _ in range(n):
        cmd, cs = struct.unpack_from('<II', b, o)
        if cmd == 0x19:
            name = b[o+8:o+24].split(b'\0')[0].decode(); vm, vs, fo, fs = struct.unpack_from('<4Q', b, o+24)
            seg[name] = (fo, fs); ns = struct.unpack_from('<I', b, o+64)[0]; so = o+72
            for _ in range(ns):
                sn = b[so:so+16].split(b'\0')[0].decode(); size = struct.unpack_from('<Q', b, so+40)[0]
                sect[(name, sn)] = (struct.unpack_from('<I', b, so+48)[0], size); so += 80
        o += cs
    return seg, sect

# ----------------------------------------------------------------------------- backbone (attention + dense FFN)
class Backbone:
    """2-bit weights in segment __KERN_3. Each projection is split over the 16 ANE units; a unit = 64 B shared codebook
    then per bank: 32 B (16 fp16 row scales) + 4*in B codes (input-major, lane j in bits 2j..2j+1).
    Output row of (unit t, bank b, lane l) = 256*b + 16*t + l; a split chunk with more banks holds the lower banks."""
    KEYS = {'dense': {(1536, 96): 1, (1536, 160): 3, (2048, 96): 1, (1536, 32): 2, (3072, 80): 1, (3072, 16): 1},
            'sparse_std': {(1536, 96): 1, (1536, 160): 1, (2048, 96): 1},
            'kv_reuse': {(1536, 128): 1, (2048, 96): 1}}
    def __init__(self, path, seg):
        self.d = np.memmap(path, np.uint8, 'r'); fo, fs = seg['__KERN_3']; self.units = []
        a = np.asarray(self.d[fo:fo+fs//64*64]).reshape(-1, 64)
        with np.errstate(invalid='ignore'):
            v = np.frombuffer(a[:, :8].tobytes(), '<f2').astype(np.float32).reshape(-1, 4)
            cand = np.flatnonzero((a[:, 8:] == 0).all(1) & (v == PAL).all(1))
        def scale_ok(off):
            s = np.frombuffer(self.d[off:off+32].tobytes(), '<f2').astype(np.float32)
            return np.isfinite(s).all() and (s > 1e-7).all() and (s < 8).all()
        for k, ci in enumerate(cand):
            st = 64*int(ci); end = 64*int(cand[k+1]) if k+1 < cand.size else fs; best = None
            for inw in (1536, 2048, 3072):
                stride = 32+4*inw; nb = 0
                while st+64+(nb+1)*stride <= end and scale_ok(fo+st+64+nb*stride): nb += 1
                if nb and (best is None or nb*stride > best[0]*best[1]): best = (stride, nb)
            if best is None: raise RuntimeError(f'__KERN_3 unit @+{st:#x}: no valid bank layout')
            self.units.append((fo+st, best[0]-32, best[1]))
        assert len(self.units) == 219*16, f'{len(self.units)} weight units (expected 3504) — unsupported build'
        self.cs = []
        for i in range(0, len(self.units), 16):
            g = self.units[i:i+16]; assert all(u[1:] == g[0][1:] for u in g), 'ragged chunk-set'
            self.cs.append(((g[0][1]//4, g[0][2]*16), list(range(i, i+16))))
        plan = [('dense', i) for i in range(NDENSE)]+[('sparse_std', i) for i in range(NSTD)]+[('kv_reuse', i) for i in range(NREUSE)]
        rem = [dict(self.KEYS[t]) for t, _ in plan]; self.slots = [{k: [] for k in self.KEYS[t]} for t, _ in plan]; L = 0
        for ci, (key, _) in enumerate(self.cs):
            while L < len(plan) and all(v == 0 for v in rem[L].values()): L += 1
            if L == len(plan) or rem[L].get(key, 0) == 0: raise RuntimeError(f'chunk-set {ci} {key} does not fit layer {L}')
            rem[L][key] -= 1; self.slots[L][key].append(ci)
        assert all(all(v == 0 for v in r.values()) for r in rem)
        self.kind = [t for t, _ in plan]                      # depth order == file order: dense, sparse_std, kv_reuse
    def _unit(self, ui):
        off, pw, nb = self.units[ui]; I = pw//4; c = np.empty((nb, 16, I), np.uint8); s = np.empty((nb, 16), np.float16)
        for b in range(nb):
            o = off+64+b*(32+pw); s[b] = np.frombuffer(self.d[o:o+32].tobytes(), '<f2')
            x = np.frombuffer(self.d[o+32:o+32+pw].tobytes(), np.uint8)
            c[b] = np.stack([(x >> k) & 3 for k in (0, 2, 4, 6)], 1).reshape(I, 16).T
        return c, s
    def blob(self, L, keys):
        parts = []
        for key, idx in keys:
            ci = self.slots[L][key][idx]; u = [self._unit(i) for i in self.cs[ci][1]]
            parts.append((np.stack([x[0] for x in u]), np.stack([x[1] for x in u])))   # (16 units, nb, 16, in), (16, nb, 16)
        parts.sort(key=lambda p: -p[0].shape[1])
        c = np.concatenate([p[0] for p in parts], 1); s = np.concatenate([p[1] for p in parts], 1)
        return c.transpose(1, 0, 2, 3).reshape(-1, c.shape[3]), s.transpose(1, 0, 2).reshape(-1)
    def layer(self, L):
        k = self.kind[L]; out = {}
        if k == 'kv_reuse': out['attn_q'] = self.blob(L, [((1536, 128), 0)])
        else:
            c, s = self.blob(L, [((1536, 96), 0), ((1536, 160), 0)]); assert c.shape == (4096, D)
            out['attn_q'] = (c[:2048], s[:2048]); out['attn_k'] = (c[2048:3072], s[2048:3072]); out['attn_v'] = (c[3072:], s[3072:])
        out['attn_output'] = self.blob(L, [((2048, 96), 0)])
        if k == 'dense':
            out['ffn_gate'] = self.blob(L, [((1536, 160), 1), ((1536, 32), 0)])
            out['ffn_up'] = self.blob(L, [((1536, 160), 2), ((1536, 32), 1)])
            out['ffn_down'] = self.blob(L, [((3072, 80), 0), ((3072, 16), 0)])
        return out

# ----------------------------------------------------------------------------- IFP experts
class Experts:
    """ifp_rasterized_weights.bin: header → palette, down row scales [44,1536], gate/up row scales [44,380,512],
    payload: per layer 380 experts × (gate[256,1536] | up[256,1536] | down[1536,256]), 2-bit LSB-first,
    code p of a matrix → (out = 16*(p//(16*in)) + p%16, in = (p//16) % in)."""
    def __init__(self, path):
        self.d = np.memmap(path, np.uint8, 'r'); h = bytes(self.d[:0x50])
        assert h[:4] == b'BBBB' and struct.unpack_from('<I', h, 0xc)[0] == 3, 'unexpected ifp_rasterized_weights header'
        (po, pn), (so, sn), (go, gn), (do, dn) = [struct.unpack_from('<2Q', h, o) for o in (0x10, 0x20, 0x30, 0x40)]
        assert np.array_equal(np.frombuffer(self.d[po:po+8].tobytes(), '<f2').astype(np.float32), PAL)
        self.sdn = np.frombuffer(self.d[so:so+sn].tobytes(), '<f2').reshape(NL-NDENSE, D)
        self.sgu = np.frombuffer(self.d[go:go+gn].tobytes(), '<f2').reshape(NL-NDENSE, NE, 512)
        self.pay = do; self.mat = ES*D//4; assert dn == (NL-NDENSE)*NE*3*self.mat, 'payload size mismatch'
    def codes(self, layer, e, which):
        no, ni = (ES, D) if which < 2 else (D, ES)
        o = self.pay + ((layer*NE + e)*3 + which)*self.mat; b = np.frombuffer(self.d[o:o+self.mat].tobytes(), np.uint8)
        c = np.stack([b & 3, (b >> 2) & 3, (b >> 4) & 3, (b >> 6) & 3], 1).reshape(-1)
        s = self.sdn[layer] if which == 2 else self.sgu[layer, e, which*256:(which+1)*256]
        return c.reshape(no//16, ni, 16).transpose(0, 2, 1).reshape(no, ni), np.asarray(s, np.float16)

# ----------------------------------------------------------------------------- norm gains (__TEXT,__const)
def gains(hwx, sect, ffn_check):
    """Table: [input_bias][L0..L55: q_norm(128) [+k_norm(128) for L<34] + 4x1536 gains][output gain].
    Every vector is a multiple of 64 B, so only 64-B-aligned starts are considered; among the structurally valid
    starts near the first match, the one whose post-FFN gain signs agree with the FFN weights' own neuron coupling
    is chosen (a one-vector misalignment gives ~0.6 agreement, the true start ~0.99)."""
    fo, size = sect[('__TEXT', '__const')]
    v = np.fromfile(hwx, '<f2', count=size//2, offset=fo).astype(np.float32)
    def qk(p): x = v[p:p+HD]; return x.size == HD and (x > 0).mean() >= 0.85 and 0.6 < x.mean() < 4
    plan = [L < KVSRC for L in range(NL)]
    total = sum((2 if k else 1)*HD + 4*D for k in plan)
    def chain(p0):
        p = p0
        for k in plan:
            if not qk(p) or (k != qk(p+HD)): return False
            p += (2 if k else 1)*HD + 4*D
        return p == p0 + total
    def build(p0):
        out = {'input_bias': v[p0-D:p0], 'output_norm_2': v[p0+total:p0+total+D]}; p = p0
        for L, k in enumerate(plan):
            out[f'{L}.q_norm'] = v[p:p+HD]; p += HD
            if k: out[f'{L}.k_norm'] = v[p:p+HD]; p += HD
            for j, nm in enumerate(('post_attention_norm', 'ffn_norm', 'post_ffw_norm', 'layer_output_norm')):
                out[f'{L}.{nm}'] = v[p+j*D:p+(j+1)*D]
            p += 4*D
        return out
    for p0 in range(D, v.size - total - D, 32):                      # 32 fp16 = 64 B
        if qk(p0) and qk(p0+HD) and chain(p0):
            cands = [q for q in range(max(D, p0-128), p0+160, 32) if chain(q)]
            scored = sorted(((ffn_check(build(q)), q) for q in cands), reverse=True)
            agree, best = scored[0]
            log(f'gain table at __const+{2*best:#x}; FFN-coupling sign agreement {agree:.3f} '
                f'(other candidates: {[round(a_, 3) for a_, _ in scored[1:]]})')
            assert agree > 0.9, 'gain table alignment could not be validated'
            return build(best)
    raise RuntimeError('norm-gain table not found')

# ----------------------------------------------------------------------------- embedding (main-h16g.odix)
def embedding(odix):
    """int4 text embedding, 8-token interleave (per 8 tokens: per dim 4 B = 8 nibbles, low nibble first), preceded by
    [filler][zero block 2048 rows][audio 64000 rows]; per-token fp16 scale array near the end (chosen by row-norm consistency)."""
    d = np.memmap(odix, np.uint8, 'r'); N = len(d); ROW = D//2; V = 262144
    z = (np.asarray(d[:N//16*16]).reshape(-1, 16) == 0).all(1)
    e = np.flatnonzero(np.diff(np.concatenate([[0], z.view(np.int8), [0]])))
    runs = [(s*16, (t-s)*16) for s, t in zip(e[0::2], e[1::2]) if (t-s)*16 >= 2048*ROW]
    assert runs, 'embedding anchor (zero block) not found'
    text = runs[0][0] + 2048*ROW + 64000*ROW
    ids = np.arange(0, V, 997); g, l = ids//8, ids % 8
    b = np.asarray(d[text:text+V*ROW]).reshape(V//8, D, 4)
    x = b[g, :, l//2]                                                     # (n, D) bytes holding each sampled row
    q = np.where((l % 2 == 1)[:, None], x >> 4, x & 0xF).astype(np.int8); q = np.where(q > 7, q-16, q)
    qn = np.linalg.norm(q.astype(np.float32), axis=1)
    half = N//2//2*2; vv = np.frombuffer(np.asarray(d[half:N//2*2]).tobytes(), '<f2').astype(np.float32)
    ok = np.isfinite(vv) & (vv > 1e-5) & (vv < 1); e2 = np.flatnonzero(np.diff(np.concatenate([[0], ok.view(np.int8), [0]])))
    best = None
    for s_, t_ in zip(e2[0::2], e2[1::2]):
        if t_-s_ < V: continue
        for start in (half+2*s_, half+2*(t_-V)):
            S = np.frombuffer(np.asarray(d[start:start+2*V]).tobytes(), '<f2').astype(np.float32); rn = S[ids]*qn
            err = rn.std()/max(np.median(rn), 1e-9)                  # row-norm coefficient of variation
            if best is None or err < best[0]: best = (err, start)
    assert best and best[0] < 0.6, 'embedding scale array not found (row-norm consistency check failed)'
    S = np.frombuffer(np.asarray(d[best[1]:best[1]+2*V]).tobytes(), '<f2')
    log(f'embedding: codes @{text:#x}, scales @{best[1]:#x}, row-norm cv {best[0]:.3f}')
    return b, S

def pack_q4_0_rows(b, S, r0, r1):
    """rows r0..r1 (multiple of 8) of the interleaved int4 table -> Q4_0 blocks (d = per-token scale, nibble = q+8)."""
    blk = np.asarray(b[r0//8:r1//8]).astype(np.uint8)                       # (n/8, D, 4)
    n = np.empty((blk.shape[0], D, 8), np.uint8); n[..., 0::2] = blk & 0xF; n[..., 1::2] = blk >> 4
    q = np.where(n > 7, n.astype(np.int16)-16, n).astype(np.int16)
    q = np.moveaxis(q, 2, 1).reshape(r1-r0, D)
    assert np.abs(q).max() <= 7
    return pack_q4_0(q, S[r0:r1])

def pack_q4_0(q, s):
    R, Cn = q.shape; nb = Cn//32; n = (q+8).astype(np.uint8).reshape(R, nb, 32)
    qs = n[:, :, :16] | (n[:, :, 16:] << 4)
    d = np.repeat(np.asarray(s, np.float16).view(np.uint8).reshape(R, 1, 2), nb, 1)
    return np.concatenate([d, qs], -1).reshape(R, nb*18)

def pack_q2k(codes, s):
    """codes (R, C) in 0..3, s (R,) fp16 row scales -> Q2_K blocks with sc=2, m=3, d=dmin=s/2 (exact)."""
    R, Cn = codes.shape; nb = Cn//256; c = codes.reshape(R, nb, 2, 4, 32).astype(np.uint8)
    qs = (c[:, :, :, 0] | (c[:, :, :, 1] << 2) | (c[:, :, :, 2] << 4) | (c[:, :, :, 3] << 6)).reshape(R, nb, 64)
    s = np.asarray(s, np.float16); h = (s.astype(np.float32)/2).astype(np.float16)
    assert np.array_equal(h.astype(np.float32)*2, s.astype(np.float32)), 'row scale not exactly halvable'
    dd = np.repeat(h.view(np.uint8).reshape(R, 1, 2), nb, 1)
    return np.concatenate([np.full((R, nb, 16), 0x32, np.uint8), qs, dd, dd], -1).reshape(R, nb*84)

# ----------------------------------------------------------------------------- IFP router
def router(path):
    """project_experts: W1[256,1536] b1 | (scalars) | gamma[256] | W2[44*380,256] b2, fp16, 64-B aligned."""
    a = np.frombuffer(open(path, 'rb').read()[0x39A0:], '<f2').astype(np.float32)
    n2 = (NL-NDENSE)*NE
    assert a.size == 4674176+n2+20 and np.abs(a[-20:]).max() < 1e-4, 'unexpected router layout'
    W1 = a[:393216].reshape(256, D); b1 = a[393216:393472]; gm = a[393568:393824]
    W2 = a[393856:393856+n2*256].reshape(n2, 256); b2 = a[4674176:4674176+n2]
    assert np.all(W2.reshape(NL-NDENSE, NE, 256)[:, :NMASK] == 0), 'router mask check failed'
    return W1, b1, gm, W2, b2

# ----------------------------------------------------------------------------- int8 KV calibration (task stream)
def kv_scales(hwx, sect):
    """Apple's per-layer int8 K/V cache scales, read from the full-depth ANE procedure (34 K + 35 V quantize ops)."""
    fo, size = sect[('__TEXT', '__text')]
    f = open(hwx, 'rb'); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    PE, CC = 0x1140, 0; f32 = lambda u: struct.unpack('<f', struct.pack('<I', u & 0xffffffff))[0]
    o = 0; R = {}; K, V = [], []
    while o+40 <= size:
        tid, sz16 = struct.unpack_from('<2H', mm, fo+o); tsz = sz16 & 0x7FF
        if tsz < 9 or o+tsz*4 > size:
            if len(K) == KVSRC and len(V) == KVSRC+1: break
            K, V = [], []; o += 16; R = {}; continue
        w = struct.unpack_from('<%dI' % tsz, mm, fo+o); i = 10 if (w[8] >> 1) & 1 else 9; wrote = {}
        while i < tsz:
            h = w[i]; i += 1; wa = h & 0x7FFF
            if (h >> 31) == 0:
                for j in range(((h >> 15) & 0x3F)+1):
                    if i >= tsz: break
                    R[wa+j] = w[i]; wrote[wa+j] = w[i]; i += 1
            else:
                m = (h >> 15) & 0xFFFF
                if i < tsz: R[wa] = w[i]; wrote[wa] = w[i]; i += 1
                for bb in range(16):
                    if (m >> bb) & 1 and i < tsz: R[wa+bb+1] = w[i]; wrote[wa+bb+1] = w[i]; i += 1
        if (R.get(CC, 0) & 0xff) == 0x1a:                            # int8-output task
            if (PE+4) in wrote: K.append(f32(wrote[PE+4]))
            elif (PE+2) in wrote: V.append(f32(wrote[PE+2]))
        o += (tsz*4+15) & ~15
    if not (len(K) == KVSRC and len(V) == KVSRC+1): return None
    return K + [0.0]*(NL-KVSRC), V + [0.0]*(NL-KVSRC-1)

# ----------------------------------------------------------------------------- tokenizer
def tokenizer(path):
    """Flat asset: u32 V @0x20, u8 type[V] @0x24, f32 score[V], u32 offset[V+5] (5 leading declarations), string blob."""
    raw = open(path, 'rb').read(); V = struct.unpack_from('<I', raw, 0x20)[0]
    ty = np.frombuffer(raw, np.uint8, V, 0x24).copy(); sc = np.frombuffer(raw, np.float32, V, 0x24+V).copy()
    off = np.frombuffer(raw, np.uint32, V+5, 0x24+5*V); blob = 0x24+9*V+20
    toks = []
    for i in range(V):
        o = blob+int(off[i+5]); toks.append(raw[o:raw.index(b'\0', o)])
    byte_ids = np.flatnonzero(ty == 6)
    assert len(byte_ids) == 256 and toks[byte_ids[0]] == b'<0x00>' and toks[3] == b'<unk>', 'tokenizer layout check failed'
    ty = ty.astype(np.int32); ty[4:238][ty[4:238] == 5] = 4      # named specials -> USER_DEFINED
    n = 262144                                                   # text vocabulary (= embedding rows); the rest are audio/voice tokens
    return toks[:n], sc[:n], ty[:n]

# Apple's afm_cider_v2 prompt template, as FoundationModels renders it (decoded from TokenGenerationCore's token table and
# cacheableSystemPrefix; verified against SystemLanguageModel.tokenCount and the live prompt lengths):
#   <bos><start_of_turn>system\n{default instructions}<end_of_turn>\n<start_of_turn>developer\n{instructions}\n<end_of_turn>\n
#   <start_of_turn>user\n{prompt}<end_of_turn>\n ... <start_of_turn>model\n
# The developer turn is always present (empty, with no trailing newline, when there are no instructions).
TEMPLATE = ("{%- set sys_default = SYS -%}\n"
            "{%- if messages and messages[0]['role'] in ('system', 'developer') -%}"
            "{%- set dev = messages[0]['content'] -%}{%- set msgs = messages[1:] -%}"
            "{%- else -%}{%- set dev = '' -%}{%- set msgs = messages -%}{%- endif -%}\n"
            "{{ '<start_of_turn>system\\n' + sys_default + '<end_of_turn>\\n<start_of_turn>developer\\n' + (dev + '\\n' if dev else '') + '<end_of_turn>\\n' }}\n"
            "{%- for message in msgs -%}{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else 'user') + "
            "'\\n' + message['content'] + '<end_of_turn>\\n' }}{%- endfor -%}\n"
            "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\\n' }}{%- endif -%}\n")

# ----------------------------------------------------------------------------- main

# ----------------------------------------------------------------------------- speculative-decoding draft model
# Apple pairs the model with a 9M-parameter draft ("v11-ifp-9m", asset ...fm_api_generic.draft) that proposes 7 tokens per
# step. Everything below is read from that asset's Neural Engine program (binary_0.hwx) and its program.odix:
#  * weights: segment __KERN_0, one block per projection = 16 units (one per ANE core) of [64 B: a sorted 16-entry fp16
#    codebook + 32 zero bytes][codes, input-major, rows-of-this-unit nibbles per input, low nibble first]; unit u holds
#    output rows u*rows..(u+1)*rows-1; FFN gate blocks are prefixed with the 128-B Neural Engine SiLU table
#  * norm gains: __TEXT,__const, one 0x4000-byte block per ANE procedure, addressed by the tile-source offsets
#  * embeddings: int4 codes (8-token interleave) for the 162144 non-draft then the 100000 draft tokens, one per-dimension
#    fp16 scale (the readout's constant tile), and the sorted list of the draft vocabulary's token ids
SPEC_DRAFT, DRAFT_CFG = 'com.apple.fm.language.instruct_3b.fm_api_generic.draft.generic_sparse', 'v11-ifp-9m'
_CM, _L2, _PE, _NE, _TS, _TD, _KD = 0, 0x1040, 0x1140, 0x1240, 0x1340, 0x1440, 0x1540
SILU_TAG = bytes.fromhex('1ac9007c0000007c')

def ane_tasks(blob, fo, size):
    """H16 task stream -> [(task index, registers so far, registers written by this task)]."""
    o = 0; R = {}; out = []
    while o+40 <= size:
        tid, sz16 = struct.unpack_from('<2H', blob, fo+o); tsz = sz16 & 0x7FF
        if tsz < 9 or o+tsz*4 > size: o += 16; continue
        w = struct.unpack_from('<%dI' % tsz, blob, fo+o); i = 10 if (w[8] >> 1) & 1 else 9; W = {}
        while i < tsz:
            h = w[i]; i += 1; wa = h & 0x7FFF
            if (h >> 31) == 0:
                for j in range(((h >> 15) & 0x3F)+1):
                    if i >= tsz: break
                    R[wa+j] = w[i]; W[wa+j] = w[i]; i += 1
            else:
                m = (h >> 15) & 0xFFFF
                if i < tsz: R[wa] = w[i]; W[wa] = w[i]; i += 1
                for b in range(16):
                    if (m >> b) & 1 and i < tsz: R[wa+b+1] = w[i]; W[wa+b+1] = w[i]; i += 1
        out.append((w[0] & 0xffff, dict(R), W)); o += (tsz*4+15) & ~15
    return out

def draft_model():
    a = find_asset(SPEC_DRAFT)
    if a is None: sys.exit(f'error: draft asset {SPEC_DRAFT} not found under {ASSETS}')
    root = f'{a}/.AssetData'; meta = json.load(open(f'{root}/metadata.json'))
    if meta.get('model_config') != DRAFT_CFG: sys.exit(f"error: this converter supports draft {DRAFT_CFG}, the asset is {meta.get('model_config')}")
    hwx = f'{root}/model.odixpackage/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx'; odix = f'{root}/model.odixpackage/program.odix'
    blob = open(hwx, 'rb').read(); seg, sect = macho(hwx)
    T = ane_tasks(blob, *sect[('__TEXT', '__text')]); procs = []
    for t in T:
        if t[0] == 0 or not procs: procs.append([])
        procs[-1].append(t)
    is_conv = lambda t: t[2].get(_NE) == 0x46 and (_KD+4) in t[2]
    width = lambda p: next(t[1][_CM+1] for t in p if is_conv(t))                 # tokens per call (8 or 64)
    pi = next(i for i, p in enumerate(procs) if sum(map(is_conv, p)) == 11*12 and width(p) == 8)
    P = procs[pi]; kfo, ksz = sect[('__KERN_0', '__kern_0')]; K = np.frombuffer(blob, np.uint8, ksz, kfo)
    cfo, csz = sect[('__TEXT', '__const')]; C = blob[cfo:cfo+csz]; cbase = 0x4000*pi
    gain = lambda off, n: np.frombuffer(C, '<f2', n, cbase+off).astype(np.float32)
    def block(off, out, n_in):
        rpu = out//16; pre = 128 if bytes(K[off:off+8]) == SILU_TAG else 0; per = pre+64+rpu*n_in//2; rows = []
        for u in range(16):
            b = K[off+u*per: off+(u+1)*per]; cb = b[pre:pre+32].view(np.float16)
            assert np.all(np.diff(cb.astype(np.float32)) > 0) and not b[pre+32:pre+64].any(), 'draft weight block check failed'
            c = b[pre+64:]; nib = np.stack([c & 0xF, c >> 4], 1).reshape(n_in, rpu).T; rows.append(cb[nib])
        return np.concatenate(rows, 0), pre > 0
    convs, eps, lastoff, g256, ghead = [], None, None, [], []
    for k, (_, R, W) in enumerate(P):
        if (_TS+10) in W: lastoff = W[_TS+10]
        if is_conv((0, R, W)):
            Wm, silu = block(W[_KD+4], R[_CM+7], R[_CM+3])
            convs.append(dict(W=Wm, silu=silu, out=R[_CM+7], inp=R[_CM+3], td=_TD in W, dst=R.get(_L2+19)))
        if W.get(_PE) == 0x82004 and eps is None: eps = struct.unpack('<f', struct.pack('<I', R[_PE+3]))[0]
        if W.get(_PE) == 0x80004 and R.get(_CM+3) == width(P) and R.get(_CM+2) == 1 and (_TS+1) in W: g256.append(gain(lastoff, D_DRAFT))
        if W.get(_PE) == 0x80004 and R.get(_CM+3) == 32: ghead.append(('k' if R.get(_CM+2) == 4 else 'q', gain(lastoff, 32)))
    assert len(convs) == 132 and len(g256) == 2*12+1 and len(ghead) == 24, (len(convs), len(g256), len(ghead))
    layers = []
    for L in range(12):
        c = convs[11*L:11*L+11]; heads = dict(ghead[2*L:2*L+2])
        kv = [x for x in c if x['out'] == 128]; sq = [x for x in c if x['out'] == 256 and x['inp'] == 256 and not x['silu']]
        # the FFN tiles are written to descending L2 addresses, so the 768-wide hidden is [tile 2 | tile 1 | tile 0]
        # (the last layer's buffers are re-banked and one destination register is left implicit, so use conv order)
        up = sq[2:][::-1]; gate = [x for x in c if x['silu']][::-1]
        for t_ in (up, gate):
            ds = [x['dst'] for x in t_]
            assert len(set(ds)) < 3 or ds == sorted(ds), 'draft FFN tile order check failed'
        down = [x for x in c if x['inp'] == 768]
        assert len(kv) == 2 and len(sq) == 5 and len(gate) == 3 and len(down) == 1 and sum(x['td'] for x in kv) == 1
        layers.append(dict(q=sq[0]['W'], k=next(x for x in kv if not x['td'])['W'], v=next(x for x in kv if x['td'])['W'],
                           o=sq[1]['W'], up=np.concatenate([x['W'] for x in up]), gate=np.concatenate([x['W'] for x in gate]),
                           down=down[0]['W'], attn_norm=g256[2*L], ffn_norm=g256[2*L+1], q_norm=heads['q'], k_norm=heads['k']))
    final = g256[-1]
    # embeddings (program.odix): data section = [non-draft codes][draft codes][rope table][scale tile][vocab list]
    d = np.memmap(odix, np.uint8, 'r'); data = struct.unpack_from('<Q', bytes(d[:0x18]), 0x10)[0]
    nd = sum(t[1][_CM+7] for t in procs[0] if t[2].get(_NE) == 0x10045)        # readout width = draft vocabulary size
    V = 262144; tail = np.asarray(d[len(d)-(1 << 20):]); best = None
    for al in range(4):
        u = tail[al:al+(len(tail)-al)//4*4].view('<u4').astype(np.int64); inc = np.diff(u) > 0
        e = np.flatnonzero(np.diff(np.concatenate([[0], inc.view(np.int8), [0]]))); runs = list(zip(e[0::2], e[1::2]))
        for s_, t_ in runs:
            seg_ = u[s_:t_+1]; ok = np.flatnonzero(seg_ < V)
            if len(ok) >= nd and ok[nd-1]-ok[0] == nd-1: best = seg_[ok[0]:ok[0]+nd]
    assert best is not None and best[-1] < V, 'draft vocabulary list not found'
    vocab = best
    codes = lambda X, n: np.asarray(d[X:X+n//8*1024]).reshape(n//8, D_DRAFT, 4)
    nondraft, draft = codes(data, V-nd), codes(data+(V-nd)*128, nd)
    assert bytes(d[data+V*128:data+V*128+8]) == bytes.fromhex('0100000040000000'), 'draft embedding layout check failed'
    S = np.frombuffer(C, '<f2', D_DRAFT, 0x200).astype(np.float32)              # readout constant tile = [0 | s]
    assert not np.frombuffer(C, '<f2', D_DRAFT, 0).any() and np.all(S > 0)
    r1000 = np.asarray(d[data+V*128+64+1000*64: data+V*128+64+1001*64]).view('<f2').astype(np.float32)
    theta = float((np.arccos(r1000[8])/1000)**-2)                              # cos(1000*theta_8), theta_j = base^(-j/16)
    return dict(meta=meta, layers=layers, final=final, eps=eps, S=S, vocab=vocab, nondraft=nondraft, draft=draft, theta=theta)

def write_draft(P, out):
    m = draft_model(); toks, scores, types = tokenizer(P['tok']); V = 262144
    log(f"draft {DRAFT_CFG}: 12 layers, vocabulary {len(m['vocab'])} tokens, rms eps {m['eps']:.4g}, rope base {m['theta']:.4g}")
    isd = np.zeros(V, bool); isd[m['vocab']] = True; row = np.empty(V, np.int64)
    row[m['vocab']] = np.arange(len(m['vocab'])); nd_ids = np.flatnonzero(~isd); row[nd_ids] = np.arange(len(nd_ids))
    def emb():                                                                   # every token's codes as Q4_0 with d = 1 (exact)
        out_ = np.empty((V, D_DRAFT//32*18), np.uint8)
        for r0 in range(0, V, 16384):
            ids = np.arange(r0, r0+16384); q = np.empty((16384, D_DRAFT), np.int16)
            for tab, sel in ((m['draft'], isd[ids]), (m['nondraft'], ~isd[ids])):
                rr = row[ids[sel]]; x = tab[rr//8, :, (rr % 8)//2]
                n = np.where(((rr % 8) % 2 == 1)[:, None], x >> 4, x & 0xF).astype(np.int16); q[sel] = np.where(n > 7, n-16, n)
            out_[r0:r0+16384] = pack_q4_0(q, np.ones(16384, np.float16))
        return out_
    w = gguf.GGUFWriter(path=None, arch='afm-draft')
    w.add_name(f'Apple Foundation Model {DISPLAY} draft ({DRAFT_CFG}) - local conversion')
    w.add_description('Speculative-decoding draft model converted from this Mac\'s Apple Intelligence assets by afm2gguf. '
                      'Use as llama.cpp --model-draft next to the afm GGUF; requires the `afm-draft` architecture.')
    p = 'afm-draft.'
    for k, v_ in (('context_length', int(m['meta'].get('context_length', 8192))), ('embedding_length', D_DRAFT), ('block_count', 12),
                  ('feed_forward_length', 768), ('attention.head_count', 8), ('attention.head_count_kv', 4),
                  ('attention.key_length', 32), ('attention.value_length', 32), ('rope.dimension_count', 32)):
        w.add_uint32(p+k, v_)
    w.add_float32(p+'rope.freq_base', float(round(m['theta'], -2))); w.add_float32(p+'attention.layer_norm_rms_epsilon', m['eps'])
    dp = m['meta'].get('default_parameters', {})
    w.add_uint32(p+'draft_steps', int(dp.get('draft_steps', 7)))
    w.add_tokenizer_model('llama'); w.add_tokenizer_pre('default')
    w.add_token_list(toks); w.add_token_scores([float(x) for x in scores]); w.add_token_types([int(x) for x in types])
    w.add_bos_token_id(2); w.add_eos_token_id(106); w.add_unk_token_id(3); w.add_pad_token_id(0)
    w.add_add_bos_token(True); w.add_add_eos_token(False); w.add_add_space_prefix(False)
    plan = [('token_embd.weight', (V, D_DRAFT//32*18), QT.Q4_0, emb),
            ('token_embd_scale.weight', (D_DRAFT,), QT.F32, lambda: m['S']),
            ('output_scale.weight', (D_DRAFT,), QT.F32, lambda: m['S']),
            ('output_norm.weight', (D_DRAFT,), QT.F32, lambda: m['final']),
            ('output.bias', (V,), QT.F32, lambda: np.where(isd, 0.0, -np.inf).astype(np.float32))]
    names = dict(attn_norm='attn_norm', ffn_norm='ffn_norm', q_norm='attn_q_norm', k_norm='attn_k_norm',
                 q='attn_q', k='attn_k', v='attn_v', o='attn_output', gate='ffn_gate', up='ffn_up', down='ffn_down')
    for L, lay in enumerate(m['layers']):
        for key, nm in names.items():
            arr = lay[key]; plan.append((f'blk.{L}.{nm}.weight', arr.shape, QT.F16 if arr.ndim == 2 else QT.F32, (lambda a_=arr: a_)))
    for name, shp, qt, _ in plan:
        if qt == QT.Q4_0: w.add_tensor_info(name, shp, np.dtype(np.uint8), int(np.prod(shp)), raw_dtype=qt)
        elif qt == QT.F16: w.add_tensor_info(name, shp, np.dtype(np.float16), int(np.prod(shp))*2)
        else: w.add_tensor_info(name, shp, np.dtype(np.float32), int(np.prod(shp))*4)
    w.write_header_to_file(path=out); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for name, shp, qt, fn in plan:
        arr = np.ascontiguousarray(fn()); arr = arr.astype(np.float16 if qt == QT.F16 else np.float32) if qt != QT.Q4_0 else arr
        assert arr.shape == tuple(shp), (name, arr.shape, shp); w.write_tensor_data(arr)
    w.close(); log(f'wrote {out} ({os.path.getsize(out)/2**20:.1f} MiB): use with  llama-server -m afm-v11.2-ifp.gguf -md {os.path.basename(out)}')

# ----------------------------------------------------------------------------- image encoder (vision tower + merger)
# Apple's image tokenizer (asset ...instruct_3b.image_encoder, ImageTokenizer) is a 24-layer ViT (d 1024, 16 heads x 64,
# SwiGLU 2816, attention inside 12x12-patch windows except the global layers 7/15/23, 2D RoPE, per-head q/k RMSNorm) and a
# 3x3 patch merger (9216 -> 4096, LayerNorm, GELU, 4096 -> 2048 -> 1536). Everything is read from the Neural Engine program
# (binary_0.hwx, procedure 0 = 1152 patches) and swift_preprocess_weights.bin (16x16 position table):
#  * int4 conv blocks = 16 units of [128-B activation table if the conv activates (NE reg1 bit 0x20000)][16-entry fp16 LUT]
#    [32 zero B][rows/16 fp16 row scales][rows/16 fp16 biases, ONLY if the conv has a bias (NE reg1 bit 0x10); effective
#    bias = stored x row scale][codes, input-major, low nibble first]. The header length therefore depends on the flags.
#  * fp16 blocks (patch embedding, last projection): rows/16 sub-tiles per unit, each [in][16]
#  * ViT projection tiles are written to descending addresses: channel order = reversed tile order
#  * norm gains: [bias | scale] constant tiles fetched by the planar engine (tile-source offset register)
#  * the compiler folded a 0.01 pre-norm scale into the last residual add (planar-engine operand scale) and into the last
#    FFN's up/down weights; it is undone here so the GGUF holds the trained model
SPEC_VISION = 'com.apple.fm.language.instruct_3b.image_encoder.generic_sparse'
SPEC_VISION_300M = 'com.apple.fm.language.instruct_300m.image_tokenizer.generic'   # the same encoder trained for the 300M model
V_GLOBAL, V_D, V_FF, V_L = (7, 15, 23), 1024, 2816, 24

def vision_model(spec=SPEC_VISION, out_dim=1536):
    a = find_asset(spec)
    if a is None: sys.exit(f'error: image encoder asset {spec} not found under {ASSETS}')
    pkg = f'{a}/.AssetData/model.odixpackage'; meta = json.load(open(f'{a}/.AssetData/metadata.json'))
    if meta.get('model_cls_name') != 'ImageTokenizer': sys.exit(f"error: unexpected image encoder asset {meta.get('model_cls_name')}")
    hwx = f'{pkg}/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx'; spw = open(f'{pkg}/swift_preprocess_weights.bin', 'rb').read()
    blob = open(hwx, 'rb').read(); seg, sect = macho(hwx)
    procs = []
    for t in ane_tasks(blob, *sect[('__TEXT', '__text')]):
        if t[0] == 0 or not procs: procs.append([])
        procs[-1].append(t)
    P = procs[0]
    KS = [np.frombuffer(blob, np.uint8, sect[(s, s.lower())][1], sect[(s, s.lower())][0]) for s in ('__KERN_0', '__KERN_1')]
    cfo, csz = sect[('__TEXT', '__const')]; C = np.frombuffer(blob, np.uint8, csz, cfo)
    def geom(R): n_in, out, U = R[_CM+3], R[_CM+7], R[_KD+25]; return n_in, out, U, out//16
    def lut_ok(K, R, off):
        n_in, out, U, rpu = geom(R); pre = 128 if R[_NE+1] & 0x20000 else 0
        if off + 16*U > len(K): return False
        for u in (0, 7, 15):
            h = K[off+u*U+pre: off+u*U+pre+64]
            with np.errstate(invalid='ignore', over='ignore'): lut = h[:32].view('<f2').astype(np.float32)
            if not (np.all(np.isfinite(lut)) and (np.all(lut == np.round(lut)) or np.all(np.diff(lut) > 0)) and np.all(np.abs(lut) <= 8) and not h[32:64].any()): return False
        return True
    # weight convs (dynamic attention matmuls, NE mode 0x10082, read their kernels from activations); dedupe by kernel address
    raw = []
    for k, (_, R, W) in enumerate(P):
        if (_KD+4) in W and R.get(_NE) != 0x10082: raw.append((k, R, W[_KD+4]))
    fp16 = lambda R: geom(R)[2] == geom(R)[3]*geom(R)[0]*2
    side = {}
    for k, R, off in raw:
        if fp16(R) or (geom(R)[0] == geom(R)[1] == 4096): continue
        a0, a1 = lut_ok(KS[0], R, off), lut_ok(KS[1], R, off)
        if a0 != a1: side[k] = 0 if a0 else 1
    split = min(k for k, s in side.items() if s == 1)
    assert all(s == (1 if k >= split else 0) for k, s in side.items()), 'image encoder kernel segment check failed'
    which = lambda k, R, off: side.get(k, 1 if (k >= split and off + 16*geom(R)[2] <= len(KS[1])) else 0)
    def block(k, R, off):
        K = KS[which(k, R, off)]; n_in, out, U, rpu = geom(R)
        if U == rpu*n_in*2:
            S = rpu//16; Wm = np.zeros((out, n_in), np.float32)
            for u in range(16):
                t = K[off+u*U: off+(u+1)*U].view('<f2').astype(np.float32).reshape(S, n_in, 16)
                for s in range(S): Wm[s*(out//S)+u*16: s*(out//S)+u*16+16] = t[s].T
            return dict(W=Wm, b=None, s=None, act=False, k=k, R=R)
        ne1 = R[_NE+1]; hasb = bool(ne1 & 0x10); pre = 128 if ne1 & 0x20000 else 0; hb = 64 + 2*rpu*(2 if hasb else 1)
        # a conv task that writes the NE bias register (exponent e in bits 16-20, value 0) scales its per-channel
        # biases by 2^(e - 32); tasks that do not write it use the stored biases as they are
        nb = P[k][2].get(_NE+3); bexp = 1.0
        if nb is not None: assert not nb & 0xffff, 'unexpected NE scalar bias'; bexp = 2.0 ** (((nb >> 16) & 0x1f) - 32)
        assert 0 <= U - (pre + hb + rpu*n_in//2) < 128, 'image encoder weight block size check failed'
        rows, bias, sc = [], [], []
        for u in range(16):
            b = K[off+u*U: off+(u+1)*U]; h = b[pre:pre+hb]; lut = h[:32].view('<f2').astype(np.float32)
            s = h[64:64+2*rpu].view('<f2').astype(np.float32); c = b[pre+hb: pre+hb+rpu*n_in//2]
            rows.append(lut[np.stack([c & 0xF, c >> 4], 1).reshape(n_in, rpu).T] * s[:, None]); sc.append(s)
            bias.append(h[64+2*rpu:64+4*rpu].view('<f2').astype(np.float32) * s * bexp if hasb else np.zeros(rpu, np.float32))
        return dict(W=np.concatenate(rows), b=np.concatenate(bias), s=np.concatenate(sc), act=pre > 0, k=k, R=R)
    seen, convs = set(), []
    for k, R, off in raw:
        key = (which(k, R, off), off)
        if key not in seen: seen.add(key); convs.append((k, R, off))
    pe = [block(*c)['W'] for c in convs[:2]]; assert pe[0].shape[1] == 588, 'patch embedding check failed'
    M = {'patch_embd': np.concatenate(pe[::-1]).reshape(V_D, 3, 14, 14)}            # tiles at descending addresses
    rest, layers, cur = convs[2:], [], []
    for c in rest:
        cur.append(c)
        if sum(1 for x in cur if x[1][_CM+3] == V_FF) == 4: layers.append(cur); cur = []
    assert len(layers) == V_L, 'image encoder layer count check failed'
    rev = lambda a, t: a.reshape(t, -1, *a.shape[1:])[::-1].reshape(a.shape)
    for L, g in enumerate(layers):
        d = [block(*c) for c in g]; a1 = [x for x in d if x['R'][_CM+3] == V_D]
        attn = [x for x in a1 if not x['act']][:16]; ids = {id(x) for x in attn}; mlp = [x for x in a1 if id(x) not in ids]
        cat = lambda xs, f: np.concatenate([x[f] for x in xs])
        grp = [(rev(cat(attn[4*i:4*i+4], 'W'), 4), rev(cat(attn[4*i:4*i+4], 'b'), 4)) for i in range(4)]
        vi = 0 if (L == 0 or L in V_GLOBAL) else 1; qi, ki = [i for i in range(3) if i != vi]
        if L == 0: qi, ki = ki, qi        # layer 0 emits K before Q (checked against Apple's own encoder: 0.85-0.97 -> 0.98-0.99)
        M[L] = dict(q=grp[qi], k=grp[ki], v=grp[vi], o=grp[3],
                    gate=rev(cat([x for x in mlp if x['act']], 'W'), 11), up=rev(cat([x for x in mlp if not x['act']], 'W'), 11),
                    down=rev(cat([x for x in d if x['R'][_CM+3] == V_FF], 'W'), 4))
    ad = [block(*c) if not (c[1][_CM+3] == c[1][_CM+7] == 4096) else c for c in cur]
    M['merge'] = np.concatenate([x['W'] for x in ad if isinstance(x, dict) and x['R'][_CM+3] == 9216])
    M['proj1'] = np.concatenate([x['W'] for x in ad if isinstance(x, dict) and x['R'][_CM+3] == 4096])
    M['proj2'] = np.concatenate([x['W'] for x in ad if isinstance(x, dict) and x['R'][_CM+3] == 2048])
    k, R, off = next(x for x in ad if not isinstance(x, dict)); U = R[_KD+25]; K = KS[which(k, R, off)]
    bg = np.stack([K[off+u*U+128: off+(u+1)*U].view('<f2').astype(np.float32).reshape(-1, 2) for u in range(16)]).reshape(-1, 2)
    M['mm_ln_b'], M['mm_ln_g'] = bg[:, 0], bg[:, 1]                                  # depthwise conv: (bias, weight) per channel
    assert M['merge'].shape == (4096, 9216) and M['proj1'].shape == (2048, 4096) and M['proj2'].shape == (out_dim, 2048) and len(bg) == 4096
    # norm gains, in program order: pre-norm, (attention, FFN) x 24, final; q/k head gains (q, k) x 24
    g1, g64, fold, eps = [], [], [], None
    for _, R, W in P:
        if W.get(_PE) == 0x80000 and (_TS+10) in W and R.get(_CM+3) in (V_D, 64) and W[_TS+10] + 8*R[_CM+3] <= csz:
            n = R[_CM+3]; t = C[W[_TS+10]: W[_TS+10]+4*n].view('<f2').astype(np.float32)
            assert not t[:n].any(), 'image encoder norm bias check failed'; (g1 if n == V_D else g64).append(t[n:])
        if W.get(_PE) == 0x80000 and (_PE+4) in W:
            s = struct.unpack('<f', struct.pack('<I', W[_PE+4]))[0]
            if abs(s - 0.125) > 1e-6: fold.append(s)
        if W.get(_PE) == 0x82004 and (_PE+3) in W and eps is None: eps = struct.unpack('<f', struct.pack('<I', W[_PE+3]))[0]
    assert len(g1) == 2*V_L+2 and len(g64) == 2*V_L and len(fold) <= 1, ('image encoder gain check failed', len(g1), len(g64), fold)
    # the final gain is ~0 on the few dimensions that carry the last layer's very large activations; those entries are
    # fp16 subnormals, which the Neural Engine treats as zero (matches Apple's encoder better on every image tried)
    M['pre_ln'], M['post_ln'] = g1[0], np.where(np.abs(g1[-1]) < 2.0**-14, 0, g1[-1]).astype(np.float32)
    f = float(np.sqrt(1/fold[0])) if fold else 1.0                                                    # undo the 0.01 fold: x (+) 0.01*f -> x + f
    for L in range(V_L):
        M[L].update(ln1=g1[1+2*L], ln2=g1[2+2*L], q_norm=g64[2*L], k_norm=g64[2*L+1])
    M[V_L-1]['up'] = M[V_L-1]['up']*f; M[V_L-1]['down'] = M[V_L-1]['down']*f
    M['pos'] = np.frombuffer(spw, '<f4', 16*16*V_D, 16).reshape(256, V_D)
    assert struct.unpack_from('<4I', spw, 0) == (16, 1024, 16, 481), 'image preprocessing weights check failed'
    return dict(meta=meta, M=M, eps=eps, fold=fold[0] if fold else None)

def write_vision(out, small=False):
    spec, od = (SPEC_VISION_300M, P_D) if small else (SPEC_VISION, 1536)
    m = vision_model(spec, od); M = m['M']
    log(f"image encoder{' (300M model)' if small else ''}: ViT {V_L} layers d{V_D}, output {od}, global layers {V_GLOBAL}, rms eps {m['eps']:.4g}, "
        + (f"folded residual scale {m['fold']:.4g} undone" if m['fold'] else 'no folded scale'))
    # Apple rotates interleaved pairs (2p, 2p+1) and alternates the axes: pair 2f = y, pair 2f+1 = x, both at frequency f.
    # llama.cpp's vision RoPE rotates (j, j+32) with y on j < 16 and x on j >= 16 (frequency j mod 16): permute q/k head dims
    pj = np.array([2*j if j < 16 else 2*(j-16)+1 for j in range(32)])
    hp = np.concatenate([2*pj, 2*pj+1])
    perm = (np.arange(16)[:, None]*64 + hp[None]).reshape(-1)
    w = gguf.GGUFWriter(path=None, arch='clip')
    w.add_name(f"Apple Foundation Model {'300M' if small else DISPLAY} image encoder - local conversion")
    w.add_description("Apple Intelligence on-device image encoder converted from this Mac's assets by afm2gguf. "
                      'Use as --mmproj next to the afm GGUF; requires llama.cpp with the `afm` projector.')
    w.add_string('clip.projector_type', 'afm'); w.add_bool('clip.has_vision_encoder', True); w.add_bool('clip.use_silu', True)
    for k, v_ in (('embedding_length', V_D), ('feed_forward_length', V_FF), ('block_count', V_L), ('projection_dim', od),
                  ('attention.head_count', 16), ('image_size', 672), ('patch_size', 14), ('spatial_merge_size', 3), ('window_size', 168)):
        w.add_uint32('clip.vision.'+k, v_)
    w.add_float32('clip.vision.attention.layer_norm_epsilon', float(m['eps']))
    w.add_array('clip.vision.wa_layer_indexes', list(V_GLOBAL))
    # pixel normalisation is host-side (not in the asset); ImageNet statistics match Apple's answers best (white reads as
    # white; CLIP's statistics make it read as gray), see the paper
    w.add_array('clip.vision.image_mean', [0.485, 0.456, 0.406]); w.add_array('clip.vision.image_std', [0.229, 0.224, 0.225])
    T = [('v.patch_embd.weight', M['patch_embd'], QT.F16), ('v.position_embd.weight', M['pos'], QT.F32),
         ('v.pre_ln.weight', M['pre_ln'], QT.F32), ('v.post_ln.weight', M['post_ln'], QT.F32)]
    for L in range(V_L):
        l, p = M[L], f'v.blk.{L}.'
        T += [(p+'ln1.weight', l['ln1'], QT.F32), (p+'ln2.weight', l['ln2'], QT.F32),
              (p+'attn_q.weight', l['q'][0][perm], QT.F16), (p+'attn_q.bias', l['q'][1][perm], QT.F32),
              (p+'attn_k.weight', l['k'][0][perm], QT.F16), (p+'attn_k.bias', l['k'][1][perm], QT.F32),
              (p+'attn_v.weight', l['v'][0], QT.F16), (p+'attn_v.bias', l['v'][1], QT.F32),
              (p+'attn_out.weight', l['o'][0], QT.F16), (p+'attn_out.bias', l['o'][1], QT.F32),
              (p+'attn_q_norm.weight', l['q_norm'][hp], QT.F32), (p+'attn_k_norm.weight', l['k_norm'][hp], QT.F32),
              (p+'ffn_gate.weight', l['gate'], QT.F16), (p+'ffn_up.weight', l['up'], QT.F16), (p+'ffn_down.weight', l['down'], QT.F16)]
    T += [('mm.0.weight', M['merge'], QT.F16), ('mm.post_norm.weight', M['mm_ln_g'], QT.F32), ('mm.post_norm.bias', M['mm_ln_b'], QT.F32),
          ('mm.1.weight', M['proj1'], QT.F16), ('mm.2.weight', M['proj2'], QT.F16)]
    for name, arr, qt in T: w.add_tensor(name, np.ascontiguousarray(arr, np.float16 if qt == QT.F16 else np.float32))
    w.write_header_to_file(path=out); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    log(f"wrote {out} ({os.path.getsize(out)/2**20:.1f} MiB): use with  llama-mtmd-cli -m {'afm-300m-base.gguf' if small else 'afm-v11.2-ifp.gguf'} --mmproj {os.path.basename(out)}")

# ----------------------------------------------------------------------------- feature adapters (rank-48 LoRA)
# Apple Intelligence features ship as LoRA adapters for this backbone: lora_i{1,2,3}_r48_48_constant_data.bin, the digit
# being the IFP level the feature runs at (ifp1/2/3 = 5/14/23 experts). The file is 218 (A, B) pairs, fp16, in segment
# order dense (o, q, k, v, linear_0, linear_1, down) x12 -> kv-reuse (o, q) x21 -> standard sparse (o, q, k, v) x23;
# A is [48, in] and B is [out, 48] (row-major, natural row order), linear_0 = up, linear_1 = gate, scale 1.
# Verified on the proofreading adapter: correct outputs become 7x likelier than with the base model; every other
# layout, and scales 0.75 / 1.25, are clearly worse.
# ----------------------------------------------------------------------------- the 300M base model ("pico")
# Apple's 300M base (asset ...instruct_300m.base, "afmplus-v11.0-pico", the base of the 300M classifiers such as the safety
# guard) is a 24-layer dense transformer (d 1024, 16 q / 4 kv heads x 64, SwiGLU 3200) with the same layer as the draft
# model, so it is written with the afm-draft architecture. Read from its Neural Engine program and program.odix:
#  * int4 conv blocks = 16 units of [16-entry fp16 LUT][32 zero B][rows/16 fp16 row scales][codes, input-major][zero pad]
#    (no biases, no activation prefix); channel order = destination-address order of the blocks
#  * the two 256-wide projections: K is the one whose output is RoPE'd and head-normalised (layer 0 emits K first,
#    every later layer V first); the per-layer multiplier on the K gain is the int8 K-cache scale (the q.k matmul
#    divides it out, attention scale = 1/sqrt(64) in every layer)
#  * norm gains, the readout scale tile [0 | s] and the vocabulary mask: __TEXT,__const tiles named by the planar engine
#  * embedding: odix data section = int4 codes, 8-token interleave, per-dimension scale s; RoPE cos/sin table after it
#  * the lora_32/64 constant data is the model's default adapter; it is written separately (--pico-adapter base.default64)
SPEC_PICO, PICO_CFG = 'com.apple.fm.language.instruct_300m.base.generic', 'v11-pico'
SPEC_TOK_300M = 'com.apple.fm.language.instruct_300m.tokenizer.generic'
P_L, P_D, P_NH, P_NKV, P_HD, P_FF = 24, 1024, 16, 4, 64, 3200

def pico_model():
    a = find_asset(SPEC_PICO)
    if a is None: sys.exit(f'error: 300M base asset {SPEC_PICO} not found under {ASSETS}')
    root = f'{a}/.AssetData'; meta = json.load(open(f'{root}/metadata.json'))
    if meta.get('model_config') != PICO_CFG: sys.exit(f"error: this converter supports {PICO_CFG}, the asset is {meta.get('model_config')}")
    hwx = f'{root}/model.odixpackage/MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx'; odix = f'{root}/model.odixpackage/program.odix'
    blob = open(hwx, 'rb').read(); seg, sect = macho(hwx)
    procs = []
    for t in ane_tasks(blob, *sect[('__TEXT', '__text')]):
        if t[0] == 0 or not procs: procs.append([])
        procs[-1].append(t)
    P = min(procs, key=len)
    KS = [np.frombuffer(blob, np.uint8, sect[(s, s.lower())][1], sect[(s, s.lower())][0]) for s in ('__KERN_0', '__KERN_1')]
    cfo, csz = sect[('__TEXT', '__const')]; C = np.frombuffer(blob, np.uint8, csz, cfo)
    def header_ok(K, off, U, rpu):
        if off + 16*U > len(K): return False
        for u in (0, 7, 15):
            h = K[off+u*U: off+u*U+64+2*rpu]
            with np.errstate(invalid='ignore', over='ignore'):
                lut = h[:32].view('<f2').astype(np.float32); s = h[64:].view('<f2').astype(np.float32)
            if not (np.all(np.isfinite(lut)) and np.all(np.diff(lut) > 0) and not h[32:64].any() and np.all(s > 0) and np.all(s < 4)): return False
        return True
    convs = [(k, R, W[_KD+4]) for k, (_, R, W) in enumerate(P) if (_KD+4) in W and R.get(_NE) == 0x46 and 32 not in (R[_CM+3], R[_CM+7])]
    assert len(convs) == 40*P_L, 'pico conv count check failed'
    side = [[i for i, K in enumerate(KS) if header_ok(K, off, R[_KD+25], R[_CM+7]//16)] for _, R, off in convs]
    segs = []                                       # blocks valid in both segments take the segment of their neighbours
    for i, s_ in enumerate(side):
        segs.append(s_[0] if len(s_) == 1 else segs[-1] if segs else next(side[j][0] for j in range(i, len(side)) if len(side[j]) == 1))
    def block(R, off, sg):
        n_in, out, U = R[_CM+3], R[_CM+7], R[_KD+25]; rpu = out//16; hb = 64 + 2*rpu; K = KS[sg]
        assert not R[_NE+1] & 0x20010 and 0 <= U - (hb + rpu*n_in//2) < 128 and header_ok(K, off, U, rpu), 'pico weight block check failed'
        rows = []
        for u in range(16):
            b = K[off+u*U: off+(u+1)*U]; lut = b[:32].view('<f2').astype(np.float32); s = b[64:hb].view('<f2').astype(np.float32)
            c = b[hb: hb+rpu*n_in//2]; rows.append(lut[np.stack([c & 0xF, c >> 4], 1).reshape(n_in, rpu).T] * s[:, None])
        return np.concatenate(rows)
    def cat(idx):
        idx = sorted(idx, key=lambda i: convs[i][1][_L2+19]); return np.concatenate([block(convs[i][1], convs[i][2], segs[i]) for i in idx])
    tiles, eps = [], None
    for _, R, W in P:
        if _PE in W and (_TS+10) in W: tiles.append(W[_TS+10])
        if W.get(_PE) == 0x82004 and (_PE+3) in W and eps is None: eps = struct.unpack('<f', struct.pack('<I', W[_PE+3]))[0]
    tile = lambda off, n: C[off: off+2*n].view('<f2').astype(np.float32)
    layers = []
    for L in range(P_L):
        b0 = 40*L; base = L*0x1100
        k1, k2 = convs[b0+4][0], convs[b0+5][0]
        kfirst = any(W.get(_PE) == 0x82004 for _, _, W in P[k1+1:k2])
        ki, vi = (b0+4, b0+5) if kfirst else (b0+5, b0+4)
        assert all(o in tiles for o in (base, base+0x800, base+0x880, base+0x900)), 'pico gain tile check failed'
        layers.append(dict(q=cat(range(b0, b0+4)), k=cat([ki]), v=cat([vi]), o=cat(range(b0+6, b0+10)), gate=cat(range(b0+10, b0+23)),
                           up=cat(range(b0+23, b0+36)), down=cat(range(b0+36, b0+40)), attn_norm=tile(base, P_D),
                           q_norm=tile(base+0x800, P_HD), k_norm=tile(base+0x880, P_HD), ffn_norm=tile(base+0x900, P_D)))
    fin = P_L*0x1100; assert fin in tiles and fin+0x800 in tiles, 'pico final tile check failed'
    final, sc = tile(fin, P_D), tile(fin+0x800, 2*P_D)
    assert not sc[:P_D].any() and np.all(sc[P_D:] > 0); S = sc[P_D:]
    mask = np.concatenate([tile(fin+0x1800+i*0x20000, 65536) for i in range(4)])
    d = np.memmap(odix, np.uint8, 'r'); data = struct.unpack_from('<Q', bytes(d[:0x18]), 0x10)[0]; V = 262144
    codes = np.asarray(d[data: data+V*P_D//2]).reshape(V//8, P_D, 4)                 # 8 tokens per 4-byte cell, lanes low-nibble first
    zero = np.flatnonzero(~codes.reshape(V//8, -1).any(1))
    assert np.array_equal(np.flatnonzero(mask < -1000), np.arange(zero[0]*8, V)) if len(zero) else False, 'pico embedding / mask check failed'
    rt = np.asarray(d[data+V*P_D//2+64: data+V*P_D//2+64+2*8192*P_HD*2]).view('<f2').astype(np.float64).reshape(2, 8192, P_HD)
    assert np.all(rt[0, 0] == 1) and not rt[1, 0].any(), 'pico RoPE table check failed'
    # angle(pos p, pair j) = p * base^(-j/32): a rough fit at p = 1, then each pair read at the largest position whose
    # angle is still unwrapped (fp16 resolution is relative, so long angles pin the base to ~1e-4)
    j = np.arange(1, P_HD//2); slope = np.polyfit(j, np.log(np.arctan2(rt[1, 1, j], rt[0, 1, j])), 1)[0]
    pj = np.minimum(8191, (2.5 / np.exp(slope*j)).astype(int)); f = np.arctan2(rt[1, pj, j], rt[0, pj, j]) / pj
    theta = float(np.exp(-P_HD//2 * np.polyfit(j, np.log(f), 1)[0]))
    return dict(meta=meta, layers=layers, final=final, S=S, mask=mask, codes=codes, eps=eps, theta=theta)

def write_pico(tok, out):
    m = pico_model(); toks, scores, types = tokenizer(tok); V = 262144
    log(f"300M base {PICO_CFG}: {P_L} layers d{P_D}, rms eps {m['eps']:.4g}, rope base {m['theta']:.4g}")
    def emb():                                                                   # int4 codes as Q4_0 with d = 1 (exact)
        o_ = np.empty((V, P_D//32*18), np.uint8)
        for r0 in range(0, V, 16384):
            blk = m['codes'][r0//8:(r0+16384)//8].astype(np.uint8)
            n = np.stack([blk & 0xF, blk >> 4], -1).reshape(-1, P_D, 8).transpose(0, 2, 1).reshape(16384, P_D)
            o_[r0:r0+16384] = pack_q4_0(np.where(n > 7, n.astype(np.int16)-16, n).astype(np.int16), np.ones(16384, np.float16))
        return o_
    w = gguf.GGUFWriter(path=None, arch='afm-draft')
    w.add_name(f"Apple Foundation Model 300M base ({m['meta'].get('display_version', PICO_CFG)}) - local conversion")
    w.add_description("Apple's 300M on-device base model converted from this Mac's Apple Intelligence assets by afm2gguf. "
                      'Requires llama.cpp with the `afm-draft` architecture.')
    p = 'afm-draft.'
    for k, v_ in (('context_length', int(m['meta'].get('context_length', 8192))), ('embedding_length', P_D), ('block_count', P_L),
                  ('feed_forward_length', P_FF), ('attention.head_count', P_NH), ('attention.head_count_kv', P_NKV),
                  ('attention.key_length', P_HD), ('attention.value_length', P_HD), ('rope.dimension_count', P_HD)):
        w.add_uint32(p+k, v_)
    w.add_float32(p+'rope.freq_base', float(f"{m['theta']:.3g}")); w.add_float32(p+'attention.layer_norm_rms_epsilon', m['eps'])
    w.add_tokenizer_model('llama'); w.add_tokenizer_pre('default')
    w.add_token_list(toks); w.add_token_scores([float(x) for x in scores]); w.add_token_types([int(x) for x in types])
    w.add_bos_token_id(2); w.add_eos_token_id(106); w.add_unk_token_id(3); w.add_pad_token_id(0)
    w.add_add_bos_token(True); w.add_add_eos_token(False); w.add_add_space_prefix(False)
    # afm_cider_v2: a system message becomes the system turn followed by an (empty) developer turn, as Apple renders it
    w.add_chat_template("{%- for message in messages -%}"
                        "{%- if message['role'] == 'system' -%}{{ '<start_of_turn>system\\n' + message['content'] + "
                        "'<end_of_turn>\\n<start_of_turn>developer\\n<end_of_turn>\\n' }}"
                        "{%- else -%}{{ '<start_of_turn>' + ('model' if message['role'] == 'assistant' else 'user') + "
                        "'\\n' + message['content'] + '<end_of_turn>\\n' }}{%- endif -%}{%- endfor -%}"
                        "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\\n' }}{%- endif -%}")
    plan = [('token_embd.weight', (V, P_D//32*18), QT.Q4_0, emb),
            ('token_embd_scale.weight', (P_D,), QT.F32, lambda: m['S']),
            ('output_scale.weight', (P_D,), QT.F32, lambda: m['S']),
            ('output_norm.weight', (P_D,), QT.F32, lambda: m['final']),
            ('output.bias', (V,), QT.F32, lambda: m['mask'])]
    names = dict(attn_norm='attn_norm', ffn_norm='ffn_norm', q_norm='attn_q_norm', k_norm='attn_k_norm',
                 q='attn_q', k='attn_k', v='attn_v', o='attn_output', gate='ffn_gate', up='ffn_up', down='ffn_down')
    for L, lay in enumerate(m['layers']):
        for key, nm in names.items():
            arr = lay[key]; plan.append((f'blk.{L}.{nm}.weight', arr.shape, QT.F16 if arr.ndim == 2 else QT.F32, (lambda a_=arr: a_)))
    for name, shp, qt, _ in plan:
        if qt == QT.Q4_0: w.add_tensor_info(name, shp, np.dtype(np.uint8), int(np.prod(shp)), raw_dtype=qt)
        elif qt == QT.F16: w.add_tensor_info(name, shp, np.dtype(np.float16), int(np.prod(shp))*2)
        else: w.add_tensor_info(name, shp, np.dtype(np.float32), int(np.prod(shp))*4)
    w.write_header_to_file(path=out); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for name, shp, qt, fn in plan:
        arr = np.ascontiguousarray(fn()); arr = arr.astype(np.float16 if qt == QT.F16 else np.float32) if qt != QT.Q4_0 else arr
        assert arr.shape == tuple(shp), (name, arr.shape, shp); w.write_tensor_data(arr)
    w.close(); log(f'wrote {out} ({os.path.getsize(out)/2**20:.1f} MiB): run with  llama-cli -m {os.path.basename(out)}')

# The 300M model's adapters: its own default adapter (lora_32/lora_64 constant data in the base package) and the task
# adapters built on it (safety guard, mm_guard, classifiers, ...), each a lora_{r}_constant_data.bin of fp16 (A [r, in],
# B [out, r]) pairs. The pairs follow the package manifest's spec names in string order, so layers come as
# 0, 10..19, 1, 20..23, 2..9; per layer: attention output, q, k, v, linear_0 (= gate), linear_1 (= up), FFN output.
# The Neural Engine program scales the A product by 0.5 (written as alpha = r/2).
PREFIX_300M = 'com.apple.fm.language.instruct_300m.'
P_ROLE = {'attention_output_transform': ('attn_output', P_NH*P_HD, P_D), 'lora_0': ('attn_q', P_D, P_NH*P_HD),
          'lora_1': ('attn_k', P_D, P_NKV*P_HD), 'lora_2': ('attn_v', P_D, P_NKV*P_HD),
          'linear_0': ('ffn_gate', P_D, P_FF), 'linear_1': ('ffn_up', P_D, P_FF), 'feed_forward_output_transform': ('ffn_down', P_FF, P_D)}

def pico_adapters():
    out = {}
    for a in sorted(glob.glob(f'{ASSETS}/*.asset')):
        try: d = plistlib.load(open(f'{a}/Info.plist', 'rb'))
        except Exception: continue
        spec = (d.get('MobileAssetProperties') or {}).get('AssetSpecifier', d.get('AssetSpecifier')) or ''
        if not spec.startswith(PREFIX_300M): continue
        name = spec[len(PREFIX_300M):].rsplit('.generic', 1)[0]
        for pkg, meta in ((f'{a}/AssetData/adapter.odixpackage', f'{a}/AssetData/metadata.json'),
                          (f'{a}/.AssetData/model.odixpackage', None)):
            for f in sorted(glob.glob(f'{pkg}/MPSGraph/mpsExecutable.mpsgraphpackage/lora_*_constant_data.bin')):
                r = int(re.search(r'lora_(\d+)_constant', f).group(1))
                key = name if meta else f'{name}.default{r}'          # the base package: base.default32 / base.default64
                if r in (32, 64): out[key] = (f, meta, r)
    return out

def write_pico_adapter(name, out):
    ads = pico_adapters()
    if name not in ads: sys.exit(f'error: 300M adapter {name!r} not found; available: {", ".join(sorted(ads))}')
    path, mpath, r = ads[name]; meta = json.load(open(mpath)) if mpath else {}
    man = plistlib.load(open(os.path.join(os.path.dirname(path), 'manifest.plist'), 'rb'))['Package Version']
    specs = next(iter(man.values()))['Adapters by rank'][f'lora_{r}']
    order = []
    for s in specs:
        m_ = re.search(r'layer_(\d+)_(attention_output_transform|attention_qkv_transform_adapters_base_adapter_(lora_\d)|'
                       r'feed_forward_hidden_transform_(linear_\d)|feed_forward_output_transform)', s)
        key = (int(m_.group(1)), m_.group(3) or m_.group(4) or m_.group(2))
        if key not in order: order.append(key)
    assert len(order) == 7*P_L and sorted(order) == sorted({(L, k) for L in range(P_L) for k in P_ROLE}), '300M adapter manifest check failed'
    v = np.fromfile(path, np.float16); o = 0; pairs = []
    for L, role in order:
        nm, a, b = P_ROLE[role]
        A = v[o:o + a*r].reshape(r, a); o += a*r; B = v[o:o + b*r].reshape(b, r); o += b*r
        pairs.append((L, nm, A, B))
    if o != v.size: sys.exit(f'error: unexpected adapter size ({v.size} values, layout needs {o})')
    log(f'300M adapter {name}: rank {r}, {len(pairs)} LoRA pairs, B rms {float(np.sqrt(np.mean([np.mean(np.square(B.astype(np.float32))) for *_, B in pairs]))):.2e}')
    w = gguf.GGUFWriter(out, 'afm-draft')
    w.add_string('general.type', 'adapter'); w.add_string('adapter.type', 'lora'); w.add_float32('adapter.lora.alpha', r/2)
    w.add_string('adapter.afm.name', name)
    for key in ('systemInstructions', 'promptContentTemplates', 'user_defined_fields'):
        if meta.get(key): w.add_string('adapter.afm.' + key, json.dumps(meta[key], ensure_ascii=False))
    for L, nm, A, B in sorted(pairs, key=lambda x: x[0]):
        w.add_tensor(f'blk.{L}.{nm}.weight.lora_a', np.ascontiguousarray(A)); w.add_tensor(f'blk.{L}.{nm}.weight.lora_b', np.ascontiguousarray(B))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    si = (meta.get('systemInstructions') or {}).get('default')
    log(f'wrote {out}: use with  llama-completion -m afm-300m-base.gguf --lora {os.path.basename(out)}' + ('  (system instructions in the GGUF metadata)' if si else ''))

LORA_R = 48
LORA_GROUPS = {'dense': [('o', 2048, 1536), ('q', 1536, 2048), ('k', 1536, 1024), ('v', 1536, 1024),
                         ('l0', 1536, 3072), ('l1', 1536, 3072), ('down', 3072, 1536)],
               'reuse': [('o', 2048, 1536), ('q', 1536, 2048)],
               'std':   [('o', 2048, 1536), ('q', 1536, 2048), ('k', 1536, 1024), ('v', 1536, 1024)]}
LORA_ORDER = [('dense', NDENSE, 0), ('reuse', NREUSE, KVSRC + 1), ('std', NSTD, NDENSE)]
LORA_ROLE = {'o': 'attn_output', 'q': 'attn_q', 'k': 'attn_k', 'v': 'attn_v', 'l0': 'ffn_up', 'l1': 'ffn_gate', 'down': 'ffn_down'}
ADAPTER_PREFIX, ADAPTER_SUFFIX = 'com.apple.fm.language.instruct_3b.', '.generic_sparse'

def adapters():
    out = {}
    for a in sorted(glob.glob(f'{ASSETS}/*.asset')):
        try: d = plistlib.load(open(f'{a}/Info.plist', 'rb'))
        except Exception: continue
        spec = (d.get('MobileAssetProperties') or {}).get('AssetSpecifier', d.get('AssetSpecifier')) or ''
        if not (spec.startswith(ADAPTER_PREFIX) and spec.endswith(ADAPTER_SUFFIX)): continue
        f = glob.glob(f'{a}/**/lora_i*_r48_48_constant_data.bin', recursive=True)
        m = [x for x in (f'{a}/AssetData/metadata.json', f'{a}/.AssetData/metadata.json') if os.path.exists(x)]
        if f: out[spec[len(ADAPTER_PREFIX):-len(ADAPTER_SUFFIX)]] = (f[0], m[0] if m else None)
    return out

def write_adapter(name, out):
    ads = adapters()
    if name not in ads: sys.exit(f'error: adapter {name!r} not found; available: {", ".join(sorted(ads))}')
    path, mpath = ads[name]; meta = json.load(open(mpath)) if mpath else {}
    if meta.get('display_version', DISPLAY) != DISPLAY: sys.exit(f"error: adapter is for {meta.get('display_version')}, not {DISPLAY}")
    ifp = int(re.search(r'lora_i(\d)_r48', os.path.basename(path)).group(1))
    v = np.fromfile(path, np.float16); o = 0; pairs = []
    for g, cnt, d0 in LORA_ORDER:
        for k in range(cnt):
            for role, a, b in LORA_GROUPS[g]:
                A = v[o:o + a*LORA_R].reshape(LORA_R, a); o += a*LORA_R
                B = v[o:o + b*LORA_R].reshape(b, LORA_R); o += b*LORA_R
                pairs.append((d0 + k, role, A, B))
    if o != v.size: sys.exit(f'error: unexpected adapter size ({v.size} values, layout needs {o})')
    brms = float(np.sqrt(np.mean([np.mean(np.square(B.astype(np.float32))) for _, _, _, B in pairs])))
    log(f'adapter {name}: {len(pairs)} LoRA pairs, ifp{ifp}, B rms {brms:.2e}' + ('  (B ~ 0: an untrained, empty adapter)' if brms < 1e-4 else ''))
    w = gguf.GGUFWriter(out, 'afm')
    w.add_string('general.type', 'adapter'); w.add_string('adapter.type', 'lora'); w.add_float32('adapter.lora.alpha', float(LORA_R))
    w.add_string('adapter.afm.name', name); w.add_uint32('adapter.afm.ifp', ifp)
    for key in ('systemInstructions', 'promptContentTemplates'):
        if meta.get(key): w.add_string('adapter.afm.' + key, json.dumps(meta[key], ensure_ascii=False))
    for L, role, A, B in pairs:
        nm = f'blk.{L}.{LORA_ROLE[role]}.weight'
        w.add_tensor(nm + '.lora_a', np.ascontiguousarray(A)); w.add_tensor(nm + '.lora_b', np.ascontiguousarray(B))
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    log(f'wrote {out}: use with  llama-server -m afm-v11.2-ifp.gguf --lora {os.path.basename(out)}  (switches to ifp{ifp} automatically)')

def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('-o', '--out'); ap.add_argument('--asset'); ap.add_argument('--tokenizer')
    ap.add_argument('--list-adapters', action='store_true', help='list the Apple Intelligence feature adapters on this Mac')
    ap.add_argument('--adapter', help='write this feature adapter as a llama.cpp LoRA GGUF instead of the model')
    ap.add_argument('--no-kvq', action='store_true', help='do not store int8 K/V cache emulation scales')
    ap.add_argument('--draft', action='store_true', help="write Apple's speculative-decoding draft model instead (arch afm-draft)")
    ap.add_argument('--vision', action='store_true', help="write Apple's image encoder as a multimodal projector (--mmproj) instead")
    ap.add_argument('--pico', action='store_true', help="write Apple's 300M base model (afmplus-v11.0-pico) instead (arch afm-draft)")
    ap.add_argument('--pico-vision', action='store_true', help="write the 300M model's image encoder (instruct_300m.image_tokenizer) as a projector instead")
    ap.add_argument('--pico-adapter', help='write this 300M adapter (e.g. safety, mm_guard, base.default64) as a llama.cpp LoRA GGUF')
    a = ap.parse_args(); t0 = time.time()
    if a.list_adapters:
        for n, (p_, _) in sorted(adapters().items()): print(f'{n:44s} ifp{re.search(r"lora_i(\d)_", p_).group(1)}')
        for n, (_, _, r) in sorted(pico_adapters().items()): print(f'{n:44s} 300M, rank {r}  (--pico-adapter)')
        return
    if not a.out: ap.error('-o/--out is required')
    if a.adapter: return write_adapter(a.adapter, a.out)
    if a.pico_adapter: return write_pico_adapter(a.pico_adapter, a.out)
    if a.vision: return write_vision(a.out)
    if a.pico_vision: return write_vision(a.out, small=True)
    if a.pico:
        tok = a.tokenizer
        if tok is None:
            t = find_asset(SPEC_TOK_300M) or find_asset(SPEC_TOK); tok = f'{t}/AssetData/tokenizer' if t else None
        if not tok or not os.path.exists(tok): sys.exit(f'error: tokenizer asset {SPEC_TOK_300M} not found; pass --tokenizer')
        return write_pico(tok, a.out)
    if a.draft:
        tok = a.tokenizer
        if tok is None:
            t = find_asset(SPEC_TOK); tok = f'{t}/AssetData/tokenizer' if t else None
        if not tok or not os.path.exists(tok): sys.exit(f'error: tokenizer asset {SPEC_TOK} not found; pass --tokenizer')
        return write_draft({'tok': tok}, a.out)
    P = locate(a)
    log('model asset:', P['meta'].get('display_version'), os.path.dirname(P['hwx']))
    cfg = json.load(open(P['ifpcfg'])); c3 = cfg['configurations'][cfg['default']]
    assert cfg['num_layers'] == NL-NDENSE and cfg['expert_size'] == ES and cfg['hidden_dim'] == D and c3['active_experts'] == NSHARED
    seg, sect = macho(P['hwx'])
    bb = Backbone(P['hwx'], seg); log(f'backbone: {len(bb.units)} weight units, {len(bb.cs)} chunk-sets')
    L5 = bb.layer(5); nrm = lambda M: M/np.linalg.norm(M, axis=1, keepdims=True)
    up5 = nrm(PAL[L5['ffn_up'][0]]*L5['ffn_up'][1].astype(np.float32)[:, None])
    dn5 = nrm((PAL[L5['ffn_down'][0]]*L5['ffn_down'][1].astype(np.float32)[:, None]).T)
    coup = np.sign(np.einsum('fr,fr->r', up5, dn5))
    ffn_check = lambda G_: float(np.mean(coup == np.sign(G_['5.post_ffw_norm'])))
    ex = Experts(P['ifp']); G = gains(P['hwx'], sect, ffn_check); W1, b1, gm, W2, b2 = router(P['router'])
    emb_b, emb_s = embedding(P['odix']); toks, scores, types = tokenizer(P['tok'])
    kv = None if a.no_kvq else kv_scales(P['hwx'], sect)
    log('int8 K/V scales:', 'not stored' if kv is None else f'K {kv[0][:3]}… V {kv[1][:3]}…')
    sysmsg = (P['api_meta'].get('systemInstructions') or {}).get('default', '')
    tpl = TEMPLATE.replace('SYS', json.dumps(sysmsg, ensure_ascii=False) if sysmsg else "''")
    w = gguf.GGUFWriter(path=None, arch='afm')
    w.add_name(f"Apple Foundation Model {DISPLAY} (on-device) - local conversion")
    w.add_description('Converted from the Apple Intelligence on-device model assets of this Mac by afm2gguf. '
                      'Exact 2-bit weights (Q2_K) and int4 embedding (Q4_0). Requires llama.cpp with the `afm` architecture.')
    p = 'afm.'
    for k, v_ in (('context_length', int(P['meta'].get('context_length', 8192))), ('embedding_length', D), ('block_count', NL),
                  ('feed_forward_length', FF), ('attention.head_count', NH), ('attention.head_count_kv', NKV),
                  ('attention.key_length', HD), ('attention.value_length', HD), ('rope.dimension_count', HD),
                  ('expert_count', NE-NMASK), ('expert_used_count', NROUTED), ('expert_shared_count', NSHARED),
                  ('expert_feed_forward_length', ES), ('expert_shared_feed_forward_length', NSHARED*ES),
                  ('leading_dense_block_count', NDENSE), ('kv_reuse_first_layer', KVSRC+1)):
        w.add_uint32(p+k, v_)
    w.add_float32(p+'rope.freq_base', 500000.0); w.add_float32(p+'attention.layer_norm_rms_epsilon', 1.0014e-5)
    w.add_float32(p+'embedding_scale', 1.0)   # Apple's gather has no multiply (recorded for clarity; the llama.cpp port ignores it)
    if kv: w.add_array(p+'attention.k_quant_scale', [float(x) for x in kv[0]]); w.add_array(p+'attention.v_quant_scale', [float(x) for x in kv[1]])
    w.add_tokenizer_model('llama'); w.add_tokenizer_pre('default')
    w.add_token_list(toks); w.add_token_scores([float(x) for x in scores]); w.add_token_types([int(x) for x in types])
    w.add_bos_token_id(2); w.add_eos_token_id(106); w.add_unk_token_id(3); w.add_pad_token_id(0)
    w.add_add_bos_token(True); w.add_add_eos_token(False); w.add_add_space_prefix(False); w.add_chat_template(tpl)

    plan = []
    q2 = lambda name, R, Cn, fn: plan.append((name, (R, Cn//256*84), QT.Q2_K, fn))
    f32 = lambda name, fn, n: plan.append((name, (n,), QT.F32, fn))
    def emb():
        out = np.empty((262144, D//32*18), np.uint8)
        for r in range(0, 262144, 16384): out[r:r+16384] = pack_q4_0_rows(emb_b, emb_s, r, r+16384)
        return out
    plan.append(('token_embd.weight', (262144, D//32*18), QT.Q4_0, emb))
    f32('input_bias.weight', lambda: G['input_bias'], D)
    f32('output_norm.weight', lambda: G[f'{NL-1}.layer_output_norm'], D)
    f32('output_norm_2.weight', lambda: G['output_norm_2'], D)
    cache = {}
    def bt(L, key):
        if cache.get('L') != L: cache.clear(); cache.update(L=L, t=bb.layer(L))
        c, s = cache['t'][key]; return pack_q2k(c, s)
    ecache = {}
    def ecodes(L, which):
        k = (L, which)
        if k not in ecache:
            ecache.clear(); r = [ex.codes(L-NDENSE, e, which) for e in range(NE)]
            ecache[k] = (np.stack([x[0] for x in r]), np.stack([x[1] for x in r]))
        return ecache[k]
    def routed(L, which):
        c, s = ecodes(L, which); c, s = c[NMASK:], s[NMASK:]; n, R, Cn = c.shape
        return pack_q2k(c.reshape(n*R, Cn), s.reshape(n*R)).reshape(n, R, -1)
    def shared(L, which):
        c, s = ecodes(L, which); c, s = c[:NSHARED], s[:NSHARED]
        if which < 2: return pack_q2k(c.reshape(-1, D), s.reshape(-1))
        assert all(np.array_equal(s[0], s[i]) for i in range(NSHARED))
        return pack_q2k(np.concatenate(list(c), 1), s[0])
    W2r = W2.reshape(NL-NDENSE, NE, 256)[:, NMASK:].reshape(-1, 256); b2r = b2.reshape(NL-NDENSE, NE)[:, NMASK:].reshape(-1)
    for L in range(NL):
        b = f'blk.{L}.'
        for nm in ('post_attention_norm', 'ffn_norm', 'post_ffw_norm', 'layer_output_norm'):
            f32(b+nm+'.weight', (lambda L=L, nm=nm: G[f'{L}.{nm}']), D)
        f32(b+'attn_q_norm.weight', (lambda L=L: G[f'{L}.q_norm']), HD)
        if L < KVSRC: f32(b+'attn_k_norm.weight', (lambda L=L: G[f'{L}.k_norm']), HD)
        q2(b+'attn_q.weight', NH*HD, D, lambda L=L: bt(L, 'attn_q'))
        if L <= KVSRC:
            q2(b+'attn_k.weight', NKV*HD, D, lambda L=L: bt(L, 'attn_k')); q2(b+'attn_v.weight', NKV*HD, D, lambda L=L: bt(L, 'attn_v'))
        q2(b+'attn_output.weight', D, NH*HD, lambda L=L: bt(L, 'attn_output'))
        if L < NDENSE:
            q2(b+'ffn_gate.weight', FF, D, lambda L=L: bt(L, 'ffn_gate')); q2(b+'ffn_up.weight', FF, D, lambda L=L: bt(L, 'ffn_up'))
            q2(b+'ffn_down.weight', D, FF, lambda L=L: bt(L, 'ffn_down'))
        else:
            q2(b+'ffn_gate_shexp.weight', NSHARED*ES, D, lambda L=L: shared(L, 0))
            q2(b+'ffn_up_shexp.weight', NSHARED*ES, D, lambda L=L: shared(L, 1))
            q2(b+'ffn_down_shexp.weight', D, NSHARED*ES, lambda L=L: shared(L, 2))
            nr = NE-NMASK
            plan.append((b+'ffn_gate_exps.weight', (nr, ES, D//256*84), QT.Q2_K, lambda L=L: routed(L, 0)))
            plan.append((b+'ffn_up_exps.weight', (nr, ES, D//256*84), QT.Q2_K, lambda L=L: routed(L, 1)))
            plan.append((b+'ffn_down_exps.weight', (nr, D, ES//256*84), QT.Q2_K, lambda L=L: routed(L, 2)))
        if L == NDENSE:
            plan.append((b+'ifp_router_fc1.weight', (256, D), QT.F32, lambda: W1))
            f32(b+'ifp_router_fc1.bias', lambda: b1, 256); f32(b+'ifp_router_norm.weight', lambda: gm, 256)
            plan.append((b+'ifp_router_fc2.weight', (W2r.shape[0], 256), QT.F32, lambda: W2r))
            f32(b+'ifp_router_fc2.bias', lambda: b2r, b2r.shape[0])
    for name, shp, qt, _ in plan:
        if qt in (QT.Q2_K, QT.Q4_0): w.add_tensor_info(name, shp, np.dtype(np.uint8), int(np.prod(shp)), raw_dtype=qt)
        else: w.add_tensor_info(name, shp, np.dtype(np.float32), int(np.prod(shp))*4)
    w.write_header_to_file(path=a.out); w.write_kv_data_to_file(); w.write_ti_data_to_file()
    for i, (name, shp, qt, fn) in enumerate(plan):
        arr = np.ascontiguousarray(fn()); arr = arr.astype(np.float32) if qt == QT.F32 else arr
        assert arr.shape == tuple(shp), (name, arr.shape, shp); w.write_tensor_data(arr)
        if i % 50 == 0 or i == len(plan)-1: log(f'[{i+1}/{len(plan)}] {name}')
    w.close(); log(f'wrote {a.out} ({os.path.getsize(a.out)/2**30:.2f} GiB) in {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
