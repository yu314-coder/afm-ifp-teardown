"""QAP recovery of the O output-group permutation -- corrected objective.

The first attempt (qap_out_perm.py) was INVALID as a test: the QAP solver returned
permutations scoring BELOW the identity (+0.070 vs +0.137), i.e. it failed to find even
the trivial solution, so its "no recovery" verdict measured the optimizer, not the data.

Cause: scipy's QAP maximises trace(A P B^T P^T) -- a raw inner product -- while the
quantity of interest is the off-diagonal CORRELATION. With uncentred matrices whose
diagonal is identically 1, the objective is dominated by the diagonal and by the mean
offset, so the optimum of the surrogate is not the optimum of the target.

Fixes:
  * zero the diagonal (self-similarity is 1 by construction and carries no ordering info)
  * centre the off-diagonal to zero mean, so trace(A P B^T P^T) IS the correlation numerator
  * seed one restart with the IDENTITY, and keep the best-by-target over all restarts, so
    the reported optimum can never be worse than doing nothing
Then apply the same decisive control: a real layout convention is layer-independent.
"""
import sys, json
import numpy as np
from scipy.optimize import quadratic_assignment
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/src')
sys.path.insert(0, '/Volumes/D/github/afm-ifp-teardown/local/pico_shapes')
import pico_weights as pw

M = json.load(open(pw.MAP_PATH))
by_layer = {}
for e in M:
    if e.get('role') == 'PARTIAL_UNIT':
        continue
    by_layer.setdefault(e['layer'], {})[e['role']] = e

NL = 6
GS = 16
G = 1024 // GS


def cosmat(V):
    n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return n @ n.T


def group_dirs(W, axis, gs=GS):
    A = W if axis == 0 else W.T
    g = A.shape[0] // gs
    return A[:g * gs].reshape(g, gs, A.shape[1]).mean(axis=1)


def prep(C):
    """zero diagonal + centre off-diagonal -> QAP inner product == correlation numerator."""
    C = C.copy()
    m = ~np.eye(C.shape[0], dtype=bool)
    C[~m] = 0.0
    C[m] -= C[m].mean()
    return C


def score(A, B, p):
    m = ~np.eye(A.shape[0], dtype=bool)
    a, b = A[np.ix_(p, p)][m], B[m]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


print('decoding ...', flush=True)
W = {}
for L in range(NL):
    ent = by_layer[L]
    for role in ['O', 'gate', 'up']:
        W[(L, role)] = np.asarray(pw.decode_tensor(ent[role]), dtype=np.float32)
print('done\n', flush=True)

CO = {L: prep(cosmat(group_dirs(W[(L, 'O')], 1))) for L in range(NL)}
CG = {L: prep(cosmat(group_dirs(W[(L, 'gate')], 0))) for L in range(NL)}
CU = {L: prep(cosmat(group_dirs(W[(L, 'up')], 0))) for L in range(NL)}

ident = np.arange(G)
print('=== QAP with centred objective; identity-seeded, best-of-12 ===')
print('%-6s %-11s %-11s %-11s %-8s' % ('layer', 'identity', 'QAP fit', 'on up', 'beat id?'))
perms = {}
for L in range(NL):
    best, bestp = score(CO[L], CG[L], ident), ident
    for t in range(12):
        opts = {'maximize': True, 'P0': 'barycenter' if t == 0 else 'randomized'}
        try:
            r = quadratic_assignment(CO[L], CG[L], method='faq', options=opts)
            p = r.col_ind
            s = score(CO[L], CG[L], p)
            if s > best:
                best, bestp = s, p
        except Exception:
            pass
    perms[L] = bestp
    print('%-6d %+.4f     %+.4f     %+.4f     %s'
          % (L, score(CO[L], CG[L], ident), best, score(CO[L], CU[L], bestp),
             'yes' if best > score(CO[L], CG[L], ident) + 1e-9 else 'NO'))

print()
print('=== CROSS-LAYER CONSISTENCY (decisive) ===  chance = %.1f%%' % (100.0 / G))
ag = [float((perms[a] == perms[b]).mean()) for a in range(NL) for b in range(a + 1, NL)]
print('pairwise agreement: mean %.1f%%  max %.1f%%' % (100 * np.mean(ag), 100 * np.max(ag)))

print()
print('=== TRANSFER of pi_0 to layers not used in its fit ===')
p0 = perms[0]
base = [score(CO[L], CG[L], ident) for L in range(1, NL)]
mv = [score(CO[L], CG[L], p0) for L in range(1, NL)]
for i, L in enumerate(range(1, NL)):
    print('  L%-2d  identity %+.4f -> pi_0 %+.4f  (%+.4f)' % (L, base[i], mv[i], mv[i] - base[i]))
print('  MEAN identity %+.4f -> pi_0 %+.4f' % (np.mean(base), np.mean(mv)))
print()
print('reference: reader-to-reader agreement is +0.68 (what correct ordering approaches)')
if np.mean(mv) > np.mean(base) + 0.15 and np.mean(ag) > 0.10:
    print('VERDICT: REAL recovery')
else:
    print('VERDICT: fit does not transfer across layers -> spurious (data-limited, not solver-limited)')
