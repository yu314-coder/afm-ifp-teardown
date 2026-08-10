"""Final validation of the nano segment_0/segment_1 block layout.

Cheap stable rank via power iteration:  sr = ||W||_F^2 / sigma_max^2 .
Also: per-block scale statistics across the 8 FFN-hidden blocks, to test the gate|up split point.
"""
import struct, collections, json
import numpy as np

OUT = '/Volumes/D/github/afm-ifp-teardown/local/nano_gguf/'
HW = ('/Volumes/D/github/afm-ifp-teardown/local/nano_asset/model.odixpackage/'
      'MPSGraph/mpsExecutable.mpsgraphpackage/binary_0.hwx')
b = open(HW, 'rb').read()
d = np.frombuffer(b, dtype=np.uint8)
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
rows.sort(key=lambda r: r['sym'])
CFG = {'s': (0x2080, 1, 2048, 32), 'a': (0xa100, 5, 2048, 32),
       'e': (0xe140, 7, 2048, 32), 'd': (0xd080, 2, 6656, 0)}
PAL = np.array([-1.5, -0.5, 0.5, 1.5], np.float32)

def read_block(bi):
    r = rows[bi]; c = r['cls']; st, N, cin, _ = CFG[c]; base = r['off']; out = []
    sub = 32+16*cin//4
    for t in range(16):
        p = base+t*st
        for k in range(N):
            q0 = p+64+k*sub
            sc = np.frombuffer(b[q0:q0+32], dtype=np.float16).astype(np.float32)
            raw = d[q0+32:q0+32+16*cin//4]
            z = np.empty(raw.size*4, np.uint8)
            z[0::4] = raw & 3; z[1::4] = (raw >> 2) & 3
            z[2::4] = (raw >> 4) & 3; z[3::4] = (raw >> 6) & 3
            slot = np.arange(z.size); oo = slot % 16; ii = slot//16
            W = np.zeros((16, cin), np.float32); W[oo, ii] = PAL[z]*sc[oo]
            out.append(W)
    return out

def stable_rank(W, iters=120, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(W.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = W@v; nu = np.linalg.norm(u)
        if nu == 0:
            return 0.0
        u /= nu
        v = W.T@u; nv = np.linalg.norm(v)
        v /= nv
    return float((W.astype(np.float64)**2).sum()/nv**2)

U0 = 'sesseseeeaaeeedddd'; U1 = 'seeseeeaaeeedddd'
R0 = {'Q': [0, 1], 'K': [2], 'V': [3], 'O': [4, 5],
      'gate': [6, 7, 8, 9], 'up': [10, 11, 12, 13], 'down': [14, 15, 16, 17]}
R1 = {'Q': [0, 1], 'O': [2, 3], 'gate': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
      'down': [12, 13, 14, 15]}
units = []; pos = 0
seq = ''.join(r['cls'] for r in rows)
while pos < len(seq):
    if seq[pos:pos+18] == U0:
        units.append(('segment_0', pos)); pos += 18
    else:
        assert seq[pos:pos+16] == U1, pos
        units.append(('segment_1', pos)); pos += 16

# matched-Gaussian baselines
rng = np.random.default_rng(1)
base = {}
for shape in [(2048, 2048), (256, 2048), (6656, 2048), (2048, 6656)]:
    base[shape] = stable_rank(rng.standard_normal(shape).astype(np.float32))
print('matched-Gaussian stable ranks:', {str(k): round(v, 1) for k, v in base.items()})

SH = {'Q': (2048, 2048), 'O': (2048, 2048), 'K': (256, 2048), 'V': (256, 2048),
      'gate': (6656, 2048), 'up': (6656, 2048), 'down': (2048, 6656)}
print('\nstable rank / matched-Gaussian ratio  (<0.5 = TRAINED); one row per layer')
print('%-4s %-4s %6s %6s %6s %6s %6s %6s %6s' % ('lay', 'seg', 'Q', 'K', 'V', 'O', 'gate', 'up', 'down'))
tab = {}
for li, (kind, start) in enumerate(units):
    U, R = (U0, R0) if kind == 'segment_0' else (U1, R1)
    row = {}
    for role, rel in R.items():
        subs = []
        for bi in [start+i for i in rel]:
            subs += read_block(bi)
        W = np.concatenate(subs, axis=0)
        assert W.shape == SH[role], (li, role, W.shape)
        row[role] = stable_rank(W)/base[SH[role]]
    tab[li] = row
    if li < 3 or li in (33, 34, 35, 36) or li > 52:
        print('%-4d %-4s %6s %6s %6s %6s %6s %6s %6s' % (
            li, kind[-1], *['%.3f' % row[r] if r in row else '  -  '
                            for r in ('Q', 'K', 'V', 'O', 'gate', 'up', 'down')]))
print(' ...')
for role in ('Q', 'K', 'V', 'O', 'gate', 'up', 'down'):
    v0 = [tab[i][role] for i in range(35) if role in tab[i]]
    v1 = [tab[i][role] for i in range(35, 56) if role in tab[i]]
    f = lambda v: 'n=%2d med=%.3f max=%.3f trained=%d/%d' % (
        len(v), float(np.median(v)), max(v), sum(x < 0.5 for x in v), len(v)) if v else '-'
    print('%-5s seg0: %-42s seg1: %s' % (role, f(v0), f(v1)))

print('\nper-block statistics for the 8 FFN-hidden blocks of segment_1 layers'
      ' (split point between blocks 7 and 8 => gate | up):')
print('%-6s %-6s %s' % ('layer', 'block', 'mean|w| per block  (rel. block index 4..11)'))
for li in (35, 40, 50, 55):
    kind, start = units[li]
    vals = []
    for rel in range(4, 12):
        W = np.concatenate(read_block(start+rel), axis=0)
        vals.append(np.abs(W).mean())
    print('%-6d %-6s %s' % (li, U1[4:12], '  '.join('%.5f' % v for v in vals)))
json.dump({k: {r: round(v, 4) for r, v in vv.items()} for k, vv in tab.items()},
          open(OUT+'stable_rank.json', 'w'), indent=1)
print('\nwrote '+OUT+'stable_rank.json')
