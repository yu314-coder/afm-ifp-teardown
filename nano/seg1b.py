"""nano segment_1 layout, using SYMTAB-INDEX order (= emission order) rather than file order.

The 6 order anomalies in file-offset order all sit exactly at __KERN_N segment boundaries: the
linker back-fills the tail of a full segment with the next small 's' blocks.  Sorting by symtab
index removes every anomaly.
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
    nm = b[stroff+strx:e].decode('latin1')
    if nm.endswith('_ne_0'):
        sg, f = seg_of(nv)
        if f is not None:
            rows.append({'sym': i, 'name': nm, 'seg': sg, 'off': f})

# size = gap to next block IN THE SAME SEGMENT (file order); last in segment -> segment end
bysegoff = collections.defaultdict(list)
for r in rows:
    bysegoff[r['seg']].append(r)
for sg, lst in bysegoff.items():
    lst.sort(key=lambda r: r['off'])
    end = SEG[sg][1]+SEG[sg][2]
    for k, r in enumerate(lst):
        r['size'] = (lst[k+1]['off'] if k+1 < len(lst) else end)-r['off']

BLK = {'s': 0x20800, 'a': 0xa1000, 'd': 0xd0800, 'e': 0xe1400}
ORD = sorted(BLK.items(), key=lambda kv: -kv[1])
PAL_REF = np.array([-1.5, -0.5, 0.5, 1.5], np.float16)
CFG = {'s': (0x2080, 1, 2048, 32), 'a': (0xa100, 5, 2048, 32),
       'e': (0xe140, 7, 2048, 32), 'd': (0xd080, 2, 6656, 0)}

def classify(r):
    for c, n in BLK.items():
        if r['size'] == n:
            return c
    for c, n in ORD:
        if r['size'] >= n:
            return c
    return '?'

for r in rows:
    r['cls'] = classify(r)

# structural validation: every one of the 16 tiles of a block must start with the palette
def tiles_ok(r):
    st = CFG[r['cls']][0]
    for t in range(16):
        p = r['off']+t*st
        if not np.array_equal(np.frombuffer(b[p:p+8], dtype=np.float16), PAL_REF):
            return False
    return True

rows.sort(key=lambda r: r['sym'])
seq = ''.join(r['cls'] for r in rows)
print('blocks=%d  class histogram=%s' % (len(seq), collections.Counter(seq).most_common()))
print('expected 56 layers: e=8*56=%d d=4*56=%d a=2*56=%d s=4*35+2*21=%d'
      % (8*56, 4*56, 2*56, 4*35+2*21))
bad = [r['sym'] for r in rows if not tiles_ok(r)]
print('blocks failing the 16-tile palette-stride check: %d %s' % (len(bad), bad[:8]))

U0 = 'sesseseeeaaeeedddd'
U1 = 'seeseeeaaeeedddd'
pos = 0; units = []
while pos < len(seq):
    if seq[pos:pos+18] == U0:
        units.append(['segment_0', pos]); pos += 18
    elif seq[pos:pos+16] == U1:
        units.append(['segment_1', pos]); pos += 16
    else:
        print('MISMATCH at %d: %r' % (pos, seq[pos:pos+20])); break
print('\nsegmentation over SYMTAB order: %s ; consumed %d/%d blocks'
      % (collections.Counter(u[0] for u in units).most_common(), pos, len(seq)))
kinds = [u[0] for u in units]
print('layer kinds: %d layers, first segment_1 at layer %d, all seg_0 then all seg_1: %s'
      % (len(units), kinds.index('segment_1'),
         kinds == ['segment_0']*35+['segment_1']*21))

# ---- unique in-place deletion argument ----
cands = [(i, j) for i in range(18) for j in range(i+1, 18)
         if U0[i] == 's' and U0[j] == 's' and
         ''.join(U0[k] for k in range(18) if k not in (i, j)) == U1]
print('\nU1 == U0 minus two s-blocks, deletions that work: %s -> K,V are U0 blocks %s'
      % (cands, cands[0]))

NSUB = {c: CFG[c][1]*16 for c in CFG}
R0 = {'Q': [0, 1], 'K': [2], 'V': [3], 'O': [4, 5],
      'gate': [6, 7, 8, 9], 'up': [10, 11, 12, 13], 'down': [14, 15, 16, 17]}
R1 = {'Q': [0, 1], 'O': [2, 3], 'gate': [4, 5, 6, 7], 'up': [8, 9, 10, 11],
      'down': [12, 13, 14, 15]}
SH = {'Q': (2048, 2048), 'O': (2048, 2048), 'K': (256, 2048), 'V': (256, 2048),
      'gate': (6656, 2048), 'up': (6656, 2048), 'down': (2048, 6656)}
for nm, U, R in (('segment_0', U0, R0), ('segment_1', U1, R1)):
    print('\n%s unit = %r (%d blocks)' % (nm, U, len(U)))
    for role in ('Q', 'K', 'V', 'O', 'gate', 'up', 'down'):
        if role not in R:
            continue
        blks = R[role]
        n = sum(NSUB[U[i]] for i in blks)
        cin = CFG[U[blks[0]]][2]
        print('   %-4s blocks %-16s classes %-6s %3d sub-blocks -> (%d, %d)  %s'
              % (role, blks, ''.join(U[i] for i in blks), n, n*16, cin,
                 'OK' if (n*16, cin) == SH[role] else 'MISMATCH exp %s' % (SH[role],)))

PAL = np.array([-1.5, -0.5, 0.5, 1.5], np.float32)
def read_block(bi):
    r = rows[bi]; c = r['cls']; st, N, cin, tl = CFG[c]; base = r['off']; out = []
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

rng = np.random.default_rng(0)
print('\n%-8s %-5s %-13s %10s %10s %8s %s'
      % ('layer', 'role', 'shape', 'stable rk', 'Gaussian', 'zeros', 'verdict'))
res = {}
for li in [0, 17, 34, 35, 36, 44, 54, 55]:
    kind, start = units[li]
    U, R = (U0, R0) if kind == 'segment_0' else (U1, R1)
    for role in ('Q', 'K', 'V', 'O', 'gate', 'up', 'down'):
        if role not in R:
            continue
        subs = []
        for bi in [start+i for i in R[role]]:
            subs += read_block(bi)
        W = np.concatenate(subs, axis=0)
        s = np.linalg.svd(W.astype(np.float64), compute_uv=False)
        G = rng.standard_normal(W.shape); sg = np.linalg.svd(G, compute_uv=False)
        sr = float((s**2).sum()/s[0]**2); srg = float((sg**2).sum()/sg[0]**2)
        res['%d/%s' % (li, role)] = [sr, srg]
        print('%-8s %-5s %-13s %10.1f %10.1f %8.4f  %s%s'
              % ('%d %s' % (li, kind[-1]), role, str(W.shape), sr, srg, (W == 0).mean(),
                 'TRAINED' if sr < srg/2 else 'random-like',
                 '' if W.shape == SH[role] else '  SHAPE!=%s' % (SH[role],)))

json.dump({'U0': U0, 'U1': U1, 'R0': R0, 'R1': R1, 'kinds': kinds,
           'units': units, 'seq': seq,
           'blocks': [{'sym': r['sym'], 'off': r['off'], 'cls': r['cls'], 'seg': r['seg']}
                      for r in rows]},
          open(OUT+'layout.json', 'w'))
print('\nwrote '+OUT+'layout.json')
