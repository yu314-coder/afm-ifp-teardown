"""Search for the O/down OUTPUT permutation using the residual-basis oracle.

residual_basis_test.py localised the defect: O/down OUTPUT channels are uncorrelated
with the residual basis (+0.011 / +0.022) while two known-correct readers agree at
+0.650. The intra-bank order is likely fine (the LoRA Rosetta result, #19), so the
mis-ordering is at the BANK / BLOCK level -- the `ob + b*nout + o` global output index
in the decoder, i.e. which (block, bank) group of 16 outputs lands where.

Method:
  * group both profiles into G groups of `gs` consecutive channels
  * the permutation maximising the correlation of two 1-D vectors is rank-matching
    (sort both, pair by rank) -- so FIT on one reader by rank-matching
  * CRITICAL CONTROL: rank-matching always inflates the fitted correlation, so the
    result is meaningless unless it GENERALISES. Fit the permutation on `gate` and
    score it on HELD-OUT readers (`up`, next-layer `Q`) that never touched the fit.

Interpretation:
  held-out correlation ~ calibration (+0.65)  -> real permutation recovered
  held-out correlation ~ 0                    -> rank-matching merely overfitted;
                                                 the norm profile does not identify it
"""
import sys, json
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

NL = 6
rng = np.random.RandomState(0)


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def group(v, gs):
    return v.reshape(-1, gs).mean(axis=1)


def rank_match(src, dst):
    """permutation p such that src[p] is rank-aligned with dst (maximises correlation)."""
    p = np.empty(src.size, dtype=int)
    p[np.argsort(dst)] = np.argsort(src)
    return p


print('decoding ...', flush=True)
W = {}
for L in range(NL + 1):
    ent = by_layer[L]
    for role in ['Q', 'O', 'gate', 'up', 'down']:
        W[(L, role)] = (np.asarray(pw.decode_down(ent[role]), dtype=np.float32) if role == 'down'
                        else np.asarray(pw.decode_tensor(ent[role]), dtype=np.float32))
print('done\n', flush=True)

for gs in (16, 64, 256):
    G = 1024 // gs
    print('=== group size %d  (%d groups) ===' % (gs, G))
    print('%-6s %-12s %-12s %-12s %-12s' % ('layer', 'fit(gate)', 'held up', 'held Q(L+1)', 'rand-perm held up'))
    hu, hq, rc = [], [], []
    for L in range(NL):
        o_out = np.linalg.norm(W[(L, 'O')], axis=0)
        g_in = np.linalg.norm(W[(L, 'gate')], axis=1)
        u_in = np.linalg.norm(W[(L, 'up')], axis=1)
        q_nx = np.linalg.norm(W[(L + 1, 'Q')], axis=1)

        go, gg, gu, gq = group(o_out, gs), group(g_in, gs), group(u_in, gs), group(q_nx, gs)
        p = rank_match(go, gg)                       # FIT on gate only
        f = corr(go[p], gg)
        a = corr(go[p], gu)                          # held out
        b = corr(go[p], gq)                          # held out
        pr = rng.permutation(G)
        c = corr(go[pr], gu)                         # random-permutation baseline
        hu.append(a); hq.append(b); rc.append(c)
        print('%-6d %+.4f      %+.4f      %+.4f      %+.4f' % (L, f, a, b, c))
    print('  MEAN held-out up %+.4f | held-out Q %+.4f | random %+.4f'
          % (np.mean(hu), np.mean(hq), np.mean(rc)))
    print()
