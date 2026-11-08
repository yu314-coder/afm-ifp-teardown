#!/usr/bin/env python3
"""afm2gguf — build a llama.cpp GGUF (architecture `afm`) of Apple's on-device foundation model
(afmplus-v11.2-ifp, `SystemLanguageModel`) from the model assets already installed on THIS Mac.

Requirements: macOS 27 with the Apple Intelligence language-model assets downloaded, Python 3.9+,
numpy, gguf (`pip install numpy gguf`). The output runs with llama.cpp built with the `afm` architecture patch.

This script contains no Apple data: every weight, scale, table and string is read from the local assets.
The weights are written exactly — Apple's 2-bit codes + fp16 row scales as Q2_K (sc=2, m=3, d=dmin=s/2 ⇒
d·sc·c − dmin·m = s·(c − 1.5), bit-exact) and the int4 embedding as Q4_0 (exact).

usage: python3 afm2gguf.py -o afm-v11.2-ifp.gguf [--asset DIR] [--tokenizer FILE] [--no-kvq]
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
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('-o', '--out', required=True); ap.add_argument('--asset'); ap.add_argument('--tokenizer')
    ap.add_argument('--no-kvq', action='store_true', help='do not store int8 K/V cache emulation scales')
    a = ap.parse_args(); t0 = time.time()
    P = locate(a); log('model asset:', P['meta'].get('display_version'), os.path.dirname(P['hwx']))
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
