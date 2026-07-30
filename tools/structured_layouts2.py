"""Structured-layout search, per-CHANNEL (grouping confound removed).

structured_layouts.py grouped channels in blocks of 16 BEFORE comparing. That is a
confound: a candidate permutation changes which channels share a group, so the score
partly measured "does this permutation form coherent groups" rather than "is this the
right order". Consistent with that, its best candidate improved the second-order score
(+0.10 -> +0.28) while making the first-order score WORSE (+0.03 -> -0.02); a genuine
layout must improve both.

Here the direction-similarity matrix is built per channel (1024x1024), so a candidate
permutation is exactly a symmetric reindex C[perm][:,perm] -- no regrouping, no confound,
and it is cheap because C is computed ONCE per tensor and then only indexed.

Reference points are computed in the same metric rather than carried over.
"""
import sys, json, itertools
import numpy as np
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import pico_weights as pw

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

NL = 4
NBLK, NBANK, NO = 4, 16, 16
MASK = ~np.eye(1024, dtype=bool)


def cosmat(A):
    """A: [1024, d] rows indexed by residual position."""
    n = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    return (n @ n.T).astype(np.float32)


def oc(A, B):
    a, b = A[MASK], B[MASK]
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print('decoding ...', flush=True)
CO, CD, CG, CU, GN, ON, DN = {}, {}, {}, {}, {}, {}, {}
for L in range(NL):
    ent = by_layer[L]
    O = np.asarray(pw.decode_tensor(ent['O']), dtype=np.float32)      # [in,out]
    D = np.asarray(pw.decode_down(ent['down']), dtype=np.float32)     # [3200,1024]
    Gt = np.asarray(pw.decode_tensor(ent['gate']), dtype=np.float32)  # [in,out]
    Ut = np.asarray(pw.decode_tensor(ent['up']), dtype=np.float32)
    CO[L] = cosmat(O.T); CD[L] = cosmat(D.T)        # writers: OUTPUT axis
    CG[L] = cosmat(Gt);  CU[L] = cosmat(Ut)         # readers: INPUT axis
    ON[L] = np.linalg.norm(O, axis=0); DN[L] = np.linalg.norm(D, axis=0)
    GN[L] = np.linalg.norm(Gt, axis=1)
    print('  layer %d' % L, flush=True)

print()
ref2 = np.mean([oc(CG[L], CU[L]) for L in range(NL)])
ref1 = np.mean([corr(GN[L], np.linalg.norm(np.asarray(pw.decode_tensor(by_layer[L]['up']),
                                                      dtype=np.float32), axis=1)) for L in range(NL)])
print('REFERENCE (two known-correct readers, same metric): 2nd %+.4f | 1st %+.4f' % (ref2, ref1))
rp = np.random.RandomState(0).permutation(1024)
print('SHUFFLED control                                  : 2nd %+.4f | 1st %+.4f'
      % (np.mean([oc(CO[L][np.ix_(rp, rp)], CG[L]) for L in range(NL)]),
         np.mean([corr(ON[L][rp], GN[L]) for L in range(NL)])))
print()

d = np.arange(1024)
FIELD = {'blk': d // 256, 'bank': (d // 16) % 16, 'o': d % 16}
SIZE = {'blk': NBLK, 'bank': NBANK, 'o': NO}

res = []
for order in itertools.permutations(['blk', 'bank', 'o']):
    for rev in itertools.product([False, True], repeat=3):
        t = np.zeros(1024, dtype=np.int64); mult = 1
        for name, r in zip(reversed(order), reversed(rev)):
            f = FIELD[name]
            if r:
                f = SIZE[name] - 1 - f
            t = t + f * mult; mult *= SIZE[name]
        if len(np.unique(t)) != 1024:
            continue
        perm = np.argsort(t)
        s2 = np.mean([oc(CO[L][np.ix_(perm, perm)], CG[L]) for L in range(NL)] +
                     [oc(CD[L][np.ix_(perm, perm)], CG[L]) for L in range(NL)])
        s1 = np.mean([corr(ON[L][perm], GN[L]) for L in range(NL)] +
                     [corr(DN[L][perm], GN[L]) for L in range(NL)])
        res.append((s2, s1, '%s%s' % ('/'.join(order), ''.join('R' if r else '.' for r in rev))))

res.sort(reverse=True)
print('%-26s %-12s %s' % ('layout (slowest->fastest)', '2nd-order', '1st-order'))
for s2, s1, lbl in res[:14]:
    mk = '   <== current decode' if lbl == 'blk/bank/o...' else ''
    print('%-26s %+.4f      %+.4f%s' % (lbl, s2, s1, mk))
cur = [r for r in res if r[2] == 'blk/bank/o...'][0]
print('...')
print('%-26s %+.4f      %+.4f   <== current decode' % (cur[2], cur[0], cur[1]))
print()
print('best: %s  2nd %+.4f  1st %+.4f   (reference 2nd %+.4f / 1st %+.4f)'
      % (res[0][2], res[0][0], res[0][1], ref2, ref1))
