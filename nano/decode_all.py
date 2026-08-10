"""Decode ALL of nano's per-layer linear weights (56 layers x 7 or 5 roles) -> nano_weights.npz.

Reuses the proven codec from wf_scratch/nano_full.py and the symtab-order block layout from
verify.py / layout.json:

  tile = [palette 32B][zeros 32B] + N x ( [16 fp16 scales = 32B][payload 16*cin/4 B] ) + [tail]
  palette = fp16 [-1.5,-0.5,+0.5,+1.5]; 2-bit codes, 4/byte, LSB-first; slot map o=slot%16, i=slot//16
  classes (stride, N, cin, tail): s=(0x2080,1,2048,32) a=(0xa100,5,2048,32)
                                  e=(0xe140,7,2048,32) d=(0xd080,2,6656,0)
  blocks = 16 tiles, ordered by LC_SYMTAB INDEX (not file offset)
  seq = 35 x 'sesseseeeaaeeedddd' + 21 x 'seeseeeaaeeedddd' = 966 blocks, no remainder

Quality gate per tensor: exact shape, all finite, and stable rank ||W||_F^2/||W||_2^2 far below
the matched-Gaussian value (ratio < 0.5).  sigma_max is EXACT here (largest eigenvalue of the
2048x2048 or 256x256 Gram matrix), not a power iteration.
"""
import struct, collections, json, zipfile, time, os
import numpy as np

OUT = '/Volumes/D/github/afm-ifp-teardown/local/nano_gguf/'
HW = ('/Volumes/D/github/afm-ifp-teardown/local/nano_asset/model.odixpackage/'
      'MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx')

t0 = time.time()
b = open(HW, 'rb').read()
d = np.frombuffer(b, dtype=np.uint8)
print('hwx %d bytes loaded in %.1fs' % (len(b), time.time()-t0))

# ---------------------------------------------------------------- Mach-O symtab / __KERN segments
nc = struct.unpack('<I', b[16:20])[0]
o = 32; symoff = nsyms = stroff = 0; SEG = {}
while nc:
    cmd, cz = struct.unpack('<II', b[o:o+8])
    if cmd == 0x2:
        symoff, nsyms, stroff, _ = struct.unpack('<4I', b[o+8:o+24])
    if cmd == 0x19:
        nm = b[o+8:o+24].rstrip(b'\0').decode('latin1')
        vm, vs, fo, fs = struct.unpack('<4Q', b[o+24:o+56])
        if nm.startswith('__KERN'):
            SEG[nm] = (vm, fo, fs)
    o += cz; nc -= 1

def seg_of(va):
    for nm, (vm, fo, fs) in SEG.items():
        if vm <= va < vm+fs:
            return nm, fo+(va-vm)
    return None, None

rows = []
for i in range(nsyms):
    s = b[symoff+i*16:symoff+i*16+16]
    strx = struct.unpack('<I', s[:4])[0]
    nv = struct.unpack('<Q', s[8:16])[0]
    e = b.index(b'\x00', stroff+strx)
    if b[stroff+strx:e].endswith(b'_ne_0'):
        sg, f = seg_of(nv)
        if f is not None:
            rows.append({'sym': i, 'seg': sg, 'off': f})

# block size = gap to the next block WITHIN the same __KERN segment (file order), last -> segment end
per = collections.defaultdict(list)
for r in rows:
    per[r['seg']].append(r)
for sg, lst in per.items():
    lst.sort(key=lambda r: r['off'])
    end = SEG[sg][1]+SEG[sg][2]
    for k, r in enumerate(lst):
        r['size'] = (lst[k+1]['off'] if k+1 < len(lst) else end)-r['off']
BLK = {'s': 0x20800, 'a': 0xa1000, 'd': 0xd0800, 'e': 0xe1400}
ORD = sorted(BLK.items(), key=lambda kv: -kv[1])
for r in rows:
    r['cls'] = next((c for c, n in BLK.items() if r['size'] == n),
                    next(c for c, n in ORD if r['size'] >= n))
rows.sort(key=lambda r: r['sym'])                 # <- symtab order is mandatory
seq = ''.join(r['cls'] for r in rows)
print('blocks: %d   class histogram %s' % (len(rows), dict(collections.Counter(seq))))
assert len(rows) == 966
assert dict(collections.Counter(seq)) == {'e': 448, 'd': 224, 's': 182, 'a': 112}

# cross-check against the recon's layout.json
LJ = json.load(open(OUT+'layout.json'))
assert LJ['seq'] == seq, 'class sequence disagrees with layout.json'
assert [r['off'] for r in rows] == [x['off'] for x in LJ['blocks']], 'block offsets disagree'
print('cross-check vs layout.json: block offsets and class sequence identical')

# ---------------------------------------------------------------- codec
CFG = {'s': (0x2080, 1, 2048, 32), 'a': (0xa100, 5, 2048, 32),
       'e': (0xe140, 7, 2048, 32), 'd': (0xd080, 2, 6656, 0)}
PAL = np.array([-1.5, -0.5, 0.5, 1.5], np.float32)
PAL_BYTES = np.array(PAL, np.float16).tobytes()
for c, (st, N, cin, tl) in CFG.items():
    assert 64+N*(32+16*cin//4)+tl == st, c

def read_block(bi):
    """Whole block -> (16*N*16, cin) float32, output rows in sub-block order."""
    r = rows[bi]; c = r['cls']; st, N, cin, _ = CFG[c]; base = r['off']
    sub = 32+16*cin//4
    nb = 4*cin                                     # payload bytes per sub-block
    # sub-block header offsets: base + t*st + 64 + k*sub   (t=0..15, k=0..N-1)
    q0 = (base+64 + np.arange(16, dtype=np.int64)[:, None]*st
          + np.arange(N, dtype=np.int64)[None, :]*sub).ravel()
    sc = d[(q0[:, None] + np.arange(32, dtype=np.int64)).ravel()]
    sc = sc.tobytes()
    sc = np.frombuffer(sc, dtype=np.float16).astype(np.float32).reshape(16*N, 16)
    raw = d[(q0[:, None] + 32 + np.arange(nb, dtype=np.int64)).ravel()].reshape(16*N, nb)
    z = np.empty((16*N, cin, 16), np.uint8)
    zf = z.reshape(16*N, nb, 4)
    zf[:, :, 0] = raw & 3
    zf[:, :, 1] = (raw >> 2) & 3
    zf[:, :, 2] = (raw >> 4) & 3
    zf[:, :, 3] = (raw >> 6) & 3
    W = PAL[z] * sc[:, None, :]                    # (16N, cin, 16) = [sub, i, o]
    return np.ascontiguousarray(W.transpose(0, 2, 1)).reshape(16*N*16, cin)

def check_palette(bi):
    r = rows[bi]; st = CFG[r['cls']][0]
    return all(b[r['off']+t*st:r['off']+t*st+8] == PAL_BYTES for t in range(16))

# reference implementation (nano_full.py / verify.py) for equivalence testing
def read_block_ref(bi):
    r = rows[bi]; c = r['cls']; st, N, cin, _ = CFG[c]; base = r['off']; out = []
    sub = 32+16*cin//4
    for t in range(16):
        p = base+t*st
        for k in range(N):
            q = p+64+k*sub
            sc = np.frombuffer(b[q:q+32], dtype=np.float16).astype(np.float32)
            raw = d[q+32:q+32+16*cin//4]
            zz = np.empty(raw.size*4, np.uint8)
            zz[0::4] = raw & 3; zz[1::4] = (raw >> 2) & 3
            zz[2::4] = (raw >> 4) & 3; zz[3::4] = (raw >> 6) & 3
            slot = np.arange(zz.size); oo = slot % 16; ii = slot//16
            W = np.zeros((16, cin), np.float32); W[oo, ii] = PAL[zz]*sc[oo]
            out.append(W)
    return np.concatenate(out, axis=0)

# equivalence: one block of each class
for c in 'saed':
    bi = seq.index(c)
    A, B = read_block(bi), read_block_ref(bi)
    assert A.shape == B.shape and np.array_equal(A, B), c
print('vectorised decoder == nano_full.py reference decoder, bit-for-bit, all 4 classes')

# structural check: every block's 16 tiles start with the fp16 palette
bad_pal = [bi for bi in range(966) if not check_palette(bi)]
print('palette-stride structural check: %d/966 blocks OK' % (966-len(bad_pal)))
assert not bad_pal, bad_pal[:10]

# ---------------------------------------------------------------- layer units and role maps
U0 = 'sesseseeeaaeeedddd'; U1 = 'seeseeeaaeeedddd'
R0 = {'Q': [0, 1], 'K': [2], 'V': [3], 'O': [4, 5],
      'gate': [6, 7, 8, 9], 'up': [10, 11, 12, 13], 'down': [14, 15, 16, 17]}
R1 = {'Q': [0, 1], 'O': [2, 3], 'gate': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
      'down': [12, 13, 14, 15]}
SH = {'Q': (2048, 2048), 'O': (2048, 2048), 'K': (256, 2048), 'V': (256, 2048),
      'gate': (6656, 2048), 'up': (6656, 2048), 'down': (2048, 6656)}
ROLES = ('Q', 'K', 'V', 'O', 'gate', 'up', 'down')
units = []; pos = 0
while pos < len(seq):
    if seq[pos:pos+18] == U0:
        units.append(('segment_0', pos)); pos += 18
    else:
        assert seq[pos:pos+16] == U1, pos
        units.append(('segment_1', pos)); pos += 16
assert len(units) == 56 and [u[0] for u in units] == ['segment_0']*35+['segment_1']*21
assert [u[1] for u in units] == [u[1] for u in LJ['units']]
print('units: %d  (35 segment_0 + 21 segment_1), consuming all 966 blocks with no remainder\n'
      % len(units))

# ---------------------------------------------------------------- stable rank (exact sigma_max)
def stable_rank(W):
    """||W||_F^2 / sigma_max^2 with sigma_max^2 = largest eigenvalue of the small-side Gram."""
    A = W.astype(np.float32)
    G = (A.T @ A) if A.shape[0] >= A.shape[1] else (A @ A.T)
    ev = np.linalg.eigvalsh(G.astype(np.float64))
    fro2 = float(np.einsum('ij,ij->', A.astype(np.float64), A.astype(np.float64)))
    return fro2/float(ev[-1]), float(np.sqrt(max(ev[-1], 0.0)))

rng = np.random.default_rng(1)
GAUSS = {}
for shape in [(2048, 2048), (256, 2048), (6656, 2048), (2048, 6656)]:
    GAUSS[shape] = stable_rank(rng.standard_normal(shape).astype(np.float32))[0]
print('matched-Gaussian stable ranks: %s' % {str(k): round(v, 1) for k, v in GAUSS.items()})

# ---------------------------------------------------------------- decode, gate, stream to npz
NPZ = OUT+'nano_weights.npz'
report = {}
fails = []
nsaved = 0
by_role = collections.Counter(); by_seg = collections.Counter()
params = 0
gmin, gmax = np.inf, -np.inf

print('\n%-4s %-4s %-5s %-12s %10s %10s %8s %8s %s' %
      ('lay', 'seg', 'role', 'shape', 'stable rk', 'gauss', 'ratio', 'max|w|', 'gate'))
zf = zipfile.ZipFile(NPZ+'.tmp', 'w', zipfile.ZIP_STORED, allowZip64=True)
for li, (kind, start) in enumerate(units):
    R = R0 if kind == 'segment_0' else R1
    for role in ROLES:
        if role not in R:
            continue
        W = np.concatenate([read_block(start+i) for i in R[role]], axis=0)
        exp = SH[role]
        ok_shape = (W.shape == exp)
        ok_finite = bool(np.isfinite(W).all())
        sr, smax = stable_rank(W)
        ratio = sr/GAUSS[exp]
        ok_rank = ratio < 0.5
        mx = float(np.abs(W).max())
        gmin = min(gmin, float(W.min())); gmax = max(gmax, float(W.max()))
        why = []
        if not ok_shape: why.append('SHAPE %s!=%s' % (W.shape, exp))
        if not ok_finite: why.append('NONFINITE')
        if not ok_rank: why.append('RANK ratio=%.3f' % ratio)
        key = '%d_%s' % (li, role)
        report[key] = {'seg': kind, 'shape': list(W.shape), 'blocks': [start+i for i in R[role]],
                       'stable_rank': round(sr, 3), 'gauss': round(GAUSS[exp], 1),
                       'ratio': round(ratio, 4), 'sigma_max': round(smax, 4),
                       'absmax': round(mx, 6), 'mean_abs': round(float(np.abs(W).mean()), 6),
                       'zero_frac': round(float((W == 0).mean()), 6),
                       'pass': not why, 'fail': why}
        if why:
            fails.append((key, kind, why))
        if li < 2 or li in (34, 35) or li == 55 or why:
            print('%-4d %-4s %-5s %-12s %10.1f %10.1f %8.3f %8.4f %s' %
                  (li, kind[-1], role, str(W.shape), sr, GAUSS[exp], ratio, mx,
                   'PASS' if not why else 'FAIL ['+'; '.join(why)+']'))
        # stream one tensor at a time so peak RAM stays ~1 tensor
        assert ok_shape and ok_finite, (key, why)      # never silently drop a tensor
        with zf.open(key+'.npy', 'w', force_zip64=True) as fh:
            np.lib.format.write_array(fh, np.ascontiguousarray(W, dtype=np.float32),
                                      allow_pickle=False)
        nsaved += 1; by_role[role] += 1; by_seg[(kind, role)] += 1
        params += W.size
        del W
zf.close()
os.replace(NPZ+'.tmp', NPZ)
print('\nwrote %s  (%d tensors, %d params, %.2f GiB, float32) in %.0fs'
      % (NPZ, nsaved, params, os.path.getsize(NPZ)/2**30, time.time()-t0))
print('global value range: [%.5f, %.5f]' % (gmin, gmax))

# ---------------------------------------------------------------- summary
print('\n=== counts by role and segment type ===')
print('%-6s %10s %10s %8s   %-14s %s' % ('role', 'segment_0', 'segment_1', 'total', 'shape', 'params'))
tot = 0
for role in ROLES:
    n0, n1 = by_seg[('segment_0', role)], by_seg[('segment_1', role)]
    p = (n0+n1)*SH[role][0]*SH[role][1]; tot += p
    print('%-6s %10d %10d %8d   %-14s %d' % (role, n0, n1, n0+n1, str(SH[role]), p))
print('%-6s %10d %10d %8d   %-14s %d' % ('TOTAL', sum(v for (k, r), v in by_seg.items()
      if k == 'segment_0'), sum(v for (k, r), v in by_seg.items() if k == 'segment_1'),
      nsaved, '', tot))
print('layers: 35 segment_0 x 7 roles = 245 ; 21 segment_1 x 5 roles = 105 ; expected total 350')

npass = sum(1 for v in report.values() if v['pass'])
print('\n=== quality gate ===')
print('tensors decoded : %d / 350' % nsaved)
print('shape exact     : %d / %d' % (sum(1 for v in report.values()
      if not any(w.startswith('SHAPE') for w in v['fail'])), nsaved))
print('all finite      : %d / %d' % (sum(1 for v in report.values()
      if not any(w == 'NONFINITE' for w in v['fail'])), nsaved))
print('trained spectrum: %d / %d' % (sum(1 for v in report.values()
      if not any(w.startswith('RANK') for w in v['fail'])), nsaved))
print('FULL PASS       : %d / %d' % (npass, nsaved))
if fails:
    print('\nFAILURES (%d) grouped by role:' % len(fails))
    g = collections.defaultdict(list)
    for key, kind, why in fails:
        g[(key.split('_', 1)[1], kind, why[0].split(' ')[0])].append(key)
    for (role, kind, w), keys in sorted(g.items()):
        rs = [report[k]['ratio'] for k in keys]
        print('  %-5s %-10s %-6s n=%3d  ratio med=%.3f min=%.3f max=%.3f  layers=%s'
              % (role, kind, w, len(keys), float(np.median(rs)), min(rs), max(rs),
                 ','.join(k.split('_')[0] for k in keys)))

print('\nper-role stable-rank ratio distribution (all 56 layers):')
for role in ROLES:
    for kind in ('segment_0', 'segment_1'):
        v = [x['ratio'] for k, x in report.items() if k.endswith('_'+role) and x['seg'] == kind]
        if v:
            print('  %-5s %-10s n=%2d  med=%.3f  min=%.3f  max=%.3f  trained(<0.5)=%d/%d'
                  % (role, kind, len(v), float(np.median(v)), min(v), max(v),
                     sum(x < 0.5 for x in v), len(v)))

json.dump(report, open(OUT+'nano_weights_report.json', 'w'), indent=1)
print('\nwrote '+OUT+'nano_weights_report.json')
